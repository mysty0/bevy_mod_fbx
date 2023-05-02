use std::{any::TypeId, path::Path};

use anyhow::{anyhow, bail, Context};
use bevy::{
    asset::{AssetLoader, BoxedFuture, LoadContext, LoadedAsset},
    math::{DVec2, DVec3, Vec2},
    prelude::{
        debug, error, info, trace, BuildWorldChildren, FromWorld, Handle, Image, Material,
        MaterialMeshBundle, Mesh, Name, Scene, StandardMaterial, Transform, TransformBundle,
        VisibilityBundle, World, WorldChildBuilder,
    },
    render::{
        mesh::{Indices, MeshVertexAttribute, PrimitiveTopology, VertexAttributeValues},
        render_resource::{AddressMode, SamplerDescriptor, VertexFormat},
        renderer::RenderDevice,
        texture::{CompressedImageFormats, ImageSampler, ImageType},
    },
    utils::HashMap,
};
use fbxcel_dom::{
    any::AnyDocument,
    v7400::{
        data::{
            mesh::layer::{tangent::Tangents, TypedLayerElementHandle},
            texture::WrapMode,
        },
        object::{
            self,
            model::{ModelHandle, TypedModelHandle},
            texture::TextureHandle,
            ObjectId, TypedObjectHandle,
        },
        Document,
    },
};

#[cfg(feature = "profile")]
use bevy::log::info_span;
use glam::{DVec4, Vec3};
use mint::Vector4;

use crate::{
    data::{FbxMesh, FbxObject, FbxScene},
    fbx_transform::FbxTransform,
    material_loader::{RawMaterialLoader, TextureLoader},
    utils::fbx_extend::{GlobalSettingsExt, ModelTreeRootExt},
    utils::triangulate,
    MaterialLoader, ATTRIBUTE_NORMAL_MAP_UV,
};

/// Bevy is kinda "meters" based while FBX (or rather: stuff exported by maya) is in "centimeters"
/// Although it doesn't mean much in practice.
const FBX_TO_BEVY_SCALE_FACTOR: f32 = 0.01;

pub struct Loader<'b, 'w, M: Material> {
    scene: FbxScene<M>,
    load_context: &'b mut LoadContext<'w>,
    suported_compressed_formats: CompressedImageFormats,
    material_loaders: Vec<&'b (dyn RawMaterialLoader<M>)>,
}

pub struct FbxLoader<'b, M: Material> {
    supported: CompressedImageFormats,
    material_loaders: Vec<&'b (dyn RawMaterialLoader<M>)>,
}
impl<'b, 'w, M: Material> FromWorld for FbxLoader<'b, M> {
    fn from_world(world: &mut World) -> Self {
        let supported = match world.get_resource::<RenderDevice>() {
            Some(render_device) => CompressedImageFormats::from_features(render_device.features()),

            None => CompressedImageFormats::all(),
        };

        let loaders: crate::FbxMaterialLoaders<M> = world.get_resource().cloned().unwrap();
        Self {
            supported,
            material_loaders: loaders.0,
        }
    }
}

impl<'w, M: Material> AssetLoader for FbxLoader<'static, M> {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let cursor = std::io::Cursor::new(bytes);
            let reader = std::io::BufReader::new(cursor);
            let maybe_doc =
                AnyDocument::from_seekable_reader(reader).expect("Failed to load document");
            if let AnyDocument::V7400(_ver, doc) = maybe_doc {
                let loader =
                    Loader::new(self.supported, self.material_loaders.clone(), load_context);
                let potential_error = loader
                    .load(*doc)
                    .await
                    .with_context(|| format!("failed to load {:?}", load_context.path()));
                if let Err(err) = potential_error {
                    error!("{err:?}");
                }
                Ok(())
            } else {
                Err(anyhow!("TODO: better error handling in fbx loader"))
            }
        })
    }
    fn extensions(&self) -> &[&str] {
        &["fbx"]
    }
}

fn spawn_scene<M: Material>(
    fbx_file_scale: f32,
    roots: &[ObjectId],
    hierarchy: &HashMap<ObjectId, FbxObject>,
    models: &HashMap<ObjectId, FbxMesh<M>>,
) -> Scene {
    #[cfg(feature = "profile")]
    let _generate_scene_span = info_span!("generate_scene").entered();

    let mut scene_world = World::default();
    scene_world
        .spawn((
            VisibilityBundle::default(),
            TransformBundle::from_transform(Transform::from_scale(
                Vec3::ONE * FBX_TO_BEVY_SCALE_FACTOR * fbx_file_scale,
            )),
            Name::new("Fbx scene root"),
        ))
        .with_children(|commands| {
            for root in roots {
                spawn_scene_rec(*root, commands, hierarchy, models);
            }
        });
    Scene::new(scene_world)
}
fn spawn_scene_rec<M: Material>(
    current: ObjectId,
    commands: &mut WorldChildBuilder,
    hierarchy: &HashMap<ObjectId, FbxObject>,
    models: &HashMap<ObjectId, FbxMesh<M>>,
) {
    let current_node = match hierarchy.get(&current) {
        Some(node) => node,
        None => return,
    };
    let mut entity = commands.spawn((
        VisibilityBundle::default(),
        TransformBundle::from_transform(current_node.transform),
    ));
    if let Some(name) = &current_node.name {
        entity.insert(Name::new(name.clone()));
    }
    entity.with_children(|commands| {
        if let Some(mesh) = models.get(&current) {
            for (mat, bevy_mesh) in mesh.materials.iter().zip(&mesh.bevy_mesh_handles) {
                let mut entity = commands.spawn(MaterialMeshBundle {
                    mesh: bevy_mesh.clone(),
                    material: mat.clone(),
                    ..Default::default()
                });
                if let Some(name) = mesh.name.as_ref() {
                    entity.insert(Name::new(name.clone()));
                }
            }
        }
        for node_id in &current_node.children {
            spawn_scene_rec(*node_id, commands, hierarchy, models);
        }
    });
}

impl<'b, 'w, M: Material> Loader<'b, 'w, M> {
    fn new(
        formats: CompressedImageFormats,
        loaders: Vec<&'b (dyn RawMaterialLoader<M>)>,
        load_context: &'b mut LoadContext<'w>,
    ) -> Self {
        Self {
            scene: FbxScene::default(),
            load_context,
            material_loaders: loaders,
            suported_compressed_formats: formats,
        }
    }

    async fn load(mut self, doc: Document) -> anyhow::Result<()> {
        info!(
            "Started loading scene {}#FbxScene",
            self.load_context.path().to_string_lossy(),
        );
        let mut meshes = HashMap::new();
        let mut hierarchy = HashMap::new();

        let fbx_scale = doc
            .global_settings()
            .and_then(|g| g.fbx_scale())
            .unwrap_or(1.0);
        let roots = doc.model_roots();
        for root in &roots {
            traverse_hierarchy(*root, &mut hierarchy);
        }

        for obj in doc.objects() {
            if let TypedObjectHandle::Model(TypedModelHandle::Mesh(mesh)) = obj.get_typed() {
                meshes.insert(obj.object_id(), self.load_mesh(mesh).await?);
            }
        }
        let roots: Vec<_> = roots.into_iter().map(|obj| obj.object_id()).collect();
        let scene = spawn_scene(fbx_scale as f32, &roots, &hierarchy, &meshes);

        let load_context = &mut self.load_context;
        load_context.set_labeled_asset("Scene", LoadedAsset::new(scene));

        let mut scene = self.scene;
        scene.hierarchy = hierarchy.clone();
        scene.roots = roots;
        load_context.set_labeled_asset("FbxScene", LoadedAsset::new(scene));
        info!(
            "Successfully loaded scene {}#FbxScene",
            load_context.path().to_string_lossy(),
        );
        Ok(())
    }

    /**
     *
    pub fn label(&self) -> String {
        return match self.name() {
            Some(name) if !name.is_empty() => format!("FbxMaterial@{name}"),
            _ => format!("FbxMaterial{}", self.object_id().raw()),
        };
    }
     */

    fn load_bevy_mesh(
        &mut self,
        mesh_obj: object::geometry::MeshHandle,
        num_materials: usize,
    ) -> anyhow::Result<Vec<Handle<Mesh>>> {
        let label = match mesh_obj.name() {
            Some(name) if !name.is_empty() => format!("FbxMesh@{name}/Primitive"),
            _ => format!("FbxMesh{}/Primitive", mesh_obj.object_id().raw()),
        };
        trace!(
            "loading geometry mesh for node_id: {:?}",
            mesh_obj.object_node_id()
        );

        #[cfg(feature = "profile")]
        let _load_geometry_mesh = info_span!("load_geometry_mesh", label = &label).entered();

        #[cfg(feature = "profile")]
        let triangulate_mesh = info_span!("traingulate_mesh", label = &label).entered();

        let polygon_vertices = mesh_obj
            .polygon_vertices()
            .context("Failed to get polygon vertices")?;
        let triangle_pvi_indices = polygon_vertices
            .triangulate_each(triangulate::triangulate)
            .context("Triangulation failed")?;

        #[cfg(feature = "profile")]
        drop(triangulate_mesh);

        // TODO this seems to duplicate vertices from neighboring triangles. We shouldn't
        // do that and instead set the indice attribute of the Mesh properly.
        let get_position = |pos: Option<_>| -> Result<_, anyhow::Error> {
            let cpi = pos.ok_or_else(|| anyhow!("Failed to get control point index"))?;
            let point = polygon_vertices
                .control_point(cpi)
                .ok_or_else(|| anyhow!("Failed to get control point: cpi={:?}", cpi))?;
            Ok(DVec3::from(point).as_vec3().into())
        };
        let positions = triangle_pvi_indices
            .iter_control_point_indices()
            .map(get_position)
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to reconstruct position vertices")?;

        debug!("Expand position lenght to {}", positions.len());

        let layer = mesh_obj
            .layers()
            .next()
            .ok_or_else(|| anyhow!("Failed to get layer"))?;

        let indices_per_material = || -> Result<_, anyhow::Error> {
            if num_materials == 0 {
                return Ok(None);
            };
            let mut indices_per_material = vec![Vec::new(); num_materials];
            let materials = layer
                .layer_element_entries()
                .find_map(|entry| match entry.typed_layer_element() {
                    Ok(TypedLayerElementHandle::Material(handle)) => Some(handle),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("Materials not found for mesh {:?}", mesh_obj))?
                .materials()
                .context("Failed to get materials")?;
            for tri_vi in triangle_pvi_indices.triangle_vertex_indices() {
                let local_material_index = materials
                    .material_index(&triangle_pvi_indices, tri_vi)
                    .context("Failed to get mesh-local material index")?
                    .to_u32();
                indices_per_material
                     .get_mut(local_material_index as usize)
                     .ok_or_else(|| {
                         anyhow!(
                             "FbxMesh-local material index out of range: num_materials={:?}, got={:?}",
                             num_materials,
                             local_material_index
                         )
                     })?
                     .push(tri_vi.to_usize() as u32);
            }
            Ok(Some(indices_per_material))
        };
        let normals = {
            let normals = layer
                .layer_element_entries()
                .find_map(|entry| match entry.typed_layer_element() {
                    Ok(TypedLayerElementHandle::Normal(handle)) => Some(handle),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("Failed to get normals"))?
                .normals()
                .context("Failed to get normals")?;
            let get_indices = |tri_vi| -> Result<_, anyhow::Error> {
                let v = normals.normal(&triangle_pvi_indices, tri_vi)?;
                Ok(DVec3::from(v).as_vec3().into())
            };
            triangle_pvi_indices
                .triangle_vertex_indices()
                .map(get_indices)
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to reconstruct normals vertices")?
        };

        let tangents = {
            let tangents =
                layer
                    .layer_element_entries()
                    .find_map(|entry| match entry.typed_layer_element() {
                        Ok(TypedLayerElementHandle::Tangent(handle)) => Some(handle),
                        _ => None,
                    });

            if let Some(tangents) = tangents {
                let tangents = tangents.tangents().context("Failed to get tangents")?;
                let get_indices = |tri_vi| -> Result<_, anyhow::Error> {
                    let v = tangents.tangent(&triangle_pvi_indices, tri_vi)?;
                    let v: Vector4<_> = [v.x, v.y, v.z, 0.0].into();
                    Ok(DVec4::from(v).as_vec4().into())
                };
                Some(
                    triangle_pvi_indices
                        .triangle_vertex_indices()
                        .map(get_indices)
                        .collect::<Result<Vec<_>, _>>()
                        .context("Failed to reconstruct tangents vertices")?,
                )
            } else {
                println!("Tangents not found in {:?}", mesh_obj.name());
                None
            }
        };

        let colors = {
            let color =
                layer
                    .layer_element_entries()
                    .find_map(|entry| match entry.typed_layer_element() {
                        Ok(TypedLayerElementHandle::Color(handle)) => Some(handle),
                        _ => None,
                    });

            if let Some(color) = color {
                let colors = color.color().context("Failed to get colors")?;

                let get_indices = |tri_vi| -> Result<_, anyhow::Error> {
                    let color = colors.color(&triangle_pvi_indices, tri_vi)?;
                    Ok(DVec4::from(color).as_vec4().into())
                };
                Some(
                    triangle_pvi_indices
                        .triangle_vertex_indices()
                        .map(get_indices)
                        .collect::<Result<Vec<_>, _>>()
                        .context("Failed to reconstruct color vertices")?,
                )
            } else {
                println!("Colors not found in {:?}", mesh_obj.name());
                None
            }
        };

        let uv = {
            let uv = layer
                .layer_element_entries()
                .find_map(|entry| match entry.typed_layer_element() {
                    Ok(TypedLayerElementHandle::Uv(handle)) => Some(handle),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("Failed to get UV"))?
                .uv()?;
            let get_indices = |tri_vi| -> Result<_, anyhow::Error> {
                let v = uv.uv(&triangle_pvi_indices, tri_vi)?;
                let fbx_uv_space = DVec2::from(v).as_vec2();
                let bevy_uv_space = fbx_uv_space * Vec2::new(1.0, -1.0) + Vec2::new(0.0, 1.0);
                Ok(bevy_uv_space.into())
            };
            triangle_pvi_indices
                .triangle_vertex_indices()
                .map(get_indices)
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to reconstruct UV vertices")?
        };

        let normal_map_uv = {
            let handle =
                layer
                    .layer_element_entries()
                    .find_map(|entry| match entry.typed_layer_element() {
                        Ok(TypedLayerElementHandle::NormalMapUv(handle)) => Some(handle),
                        _ => None,
                    });
            if let Some(handle) = handle {
                let uv = handle.uv()?;
                let get_indices = |tri_vi| -> Result<_, anyhow::Error> {
                    let v = uv.uv(&triangle_pvi_indices, tri_vi)?;
                    let fbx_uv_space = DVec2::from(v).as_vec2();
                    let bevy_uv_space = fbx_uv_space * Vec2::new(1.0, -1.0) + Vec2::new(0.0, 1.0);
                    Ok(bevy_uv_space.into())
                };
                Some(
                    triangle_pvi_indices
                        .triangle_vertex_indices()
                        .map(get_indices)
                        .collect::<Result<Vec<_>, _>>()
                        .context("Failed to reconstruct UV vertices")?,
                )
            } else {
                println!("Normal map uvs not found in {:?}", mesh_obj.name());
                None
            }
        };

        if uv.len() != positions.len() || uv.len() != normals.len() {
            bail!(
                "mismatched length of buffers: pos{} uv{} normals{}",
                positions.len(),
                uv.len(),
                normals.len(),
            );
        }

        // TODO: remove unused vertices from partial models
        // this is complicated, as it also requires updating the indices.

        // A single mesh may have multiple materials applied to a different subset of
        // its vertices. In the following code, we create a unique mesh per material
        // we found.
        let full_mesh_indices: Vec<_> = triangle_pvi_indices
            .triangle_vertex_indices()
            .map(|t| t.to_usize() as u32)
            .collect();
        let all_indices = if let Some(per_materials) = indices_per_material()? {
            per_materials
        } else {
            vec![full_mesh_indices.clone()]
        };

        debug!("Material count for {label}: {}", all_indices.len());

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(positions),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uv));

        if let Some(normal_map_uv) = normal_map_uv {
            mesh.insert_attribute(
                ATTRIBUTE_NORMAL_MAP_UV,
                VertexAttributeValues::Float32x2(normal_map_uv),
            );
        }

        if let Some(colors) = colors {
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_COLOR,
                VertexAttributeValues::Float32x4(colors),
            );
        }

        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals),
        );

        if let Some(tangents) = tangents {
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_TANGENT,
                VertexAttributeValues::Float32x4(tangents),
            );
        }

        mesh.set_indices(Some(Indices::U32(full_mesh_indices)));
        mesh.generate_tangents()
            .context("Failed to generate tangents")?;

        let all_handles = all_indices
            .into_iter()
            .enumerate()
            .map(|(i, material_indices)| {
                debug!("Material {i} has {} vertices", material_indices.len());

                let mut material_mesh = mesh.clone();
                material_mesh.set_indices(Some(Indices::U32(material_indices)));

                let label = format!("{label}{i}");

                let handle = self
                    .load_context
                    .set_labeled_asset(&label, LoadedAsset::new(material_mesh));
                self.scene.bevy_meshes.insert(handle.clone(), label);
                handle
            })
            .collect();
        Ok(all_handles)
    }

    // Note: FBX meshes can have multiple different materials, it's not just a mesh.
    // the FBX equivalent of a bevy Mesh is a geometry mesh
    async fn load_mesh(
        &mut self,
        mesh_obj: object::model::MeshHandle<'_>,
    ) -> anyhow::Result<FbxMesh<M>> {
        let label = if let Some(name) = mesh_obj.name() {
            format!("FbxMesh@{name}")
        } else {
            format!("FbxMesh{}", mesh_obj.object_id().raw())
        };
        debug!("Loading FBX mesh: {label}");

        let bevy_obj = mesh_obj.geometry().context("Failed to get geometry")?;

        // async and iterators into for are necessary because of `async` `read_asset_bytes`
        // call in `load_video_clip`  that virally infect everything.
        // This can't even be ran in parallel, because we store already-encountered materials.
        let mut materials = Vec::new();
        for mat in mesh_obj.materials() {
            let mat = self.load_material(mat).await;
            let mat = mat.context("Failed to load materials for mesh")?;
            materials.push(mat);
        }
        let material_count = materials.len();
        if material_count == 0 {
            materials.push(Handle::default());
        }

        let bevy_mesh_handles = self
            .load_bevy_mesh(bevy_obj, material_count)
            .context("Failed to load geometry mesh")?;

        let mesh = FbxMesh {
            name: mesh_obj.name().map(Into::into),
            bevy_mesh_handles,
            materials,
        };

        let mesh_handle = self
            .load_context
            .set_labeled_asset(&label, LoadedAsset::new(mesh.clone()));

        self.scene.meshes.insert(mesh_obj.object_id(), mesh_handle);

        Ok(mesh)
    }

    async fn load_raw_material(
        &mut self,
        material_obj: object::material::MaterialHandle<'_>,
    ) -> anyhow::Result<Option<M>> {
        let loaders = self.material_loaders.clone();

        for &loader in &loaders {
            let mut texture_loader = TextureLoader {
                textures: &mut self.scene.textures,
                load_context: self.load_context,
                suported_compressed_formats: self.suported_compressed_formats,
            };

            if let Some(loader_material) = loader.load(&mut texture_loader, material_obj).await? {
                return Ok(Some(loader_material));
            }
        }
        Ok(None)
    }

    async fn load_material(
        &mut self,
        material_obj: object::material::MaterialHandle<'_>,
    ) -> anyhow::Result<Handle<M>> {
        let label = match material_obj.name() {
            Some(name) if !name.is_empty() => format!("FbxMaterial@{name}"),
            _ => format!("FbxMaterial{}", material_obj.object_id().raw()),
        };
        if let Some(handle) = self.scene.materials.get(&label) {
            debug!("Already encountered material: {label}, skipping");

            return Ok(handle.clone_weak());
        }
        debug!("Loading FBX material: {label}");

        let material = self.load_raw_material(material_obj).await?;

        let material = material.context("None of the material loaders could load this material")?;
        let handle = self
            .load_context
            .set_labeled_asset(&label, LoadedAsset::new(material));
        debug!("Successfully loaded material: {label}");

        self.scene.materials.insert(label, handle.clone());
        Ok(handle)
    }
}

fn traverse_hierarchy(node: ModelHandle, hierarchy: &mut HashMap<ObjectId, FbxObject>) {
    #[cfg(feature = "profile")]
    let _hierarchy_span = info_span!("traverse_fbx_hierarchy").entered();

    traverse_hierarchy_rec(node, None, hierarchy);
    debug!("Tree has {} nodes", hierarchy.len());
    trace!("root: {:?}", node.object_node_id());
}
fn traverse_hierarchy_rec(
    node: ModelHandle,
    parent: Option<FbxTransform>,
    hierarchy: &mut HashMap<ObjectId, FbxObject>,
) -> bool {
    let name = node.name().map(|s| s.to_owned());
    let data = FbxTransform::from_node(node, parent);

    let mut mesh_leaf = false;
    node.child_models().for_each(|child| {
        mesh_leaf |= traverse_hierarchy_rec(*child, Some(data), hierarchy);
    });
    if node.subclass() == "Mesh" {
        mesh_leaf = true;
    }
    // Only keep nodes that have Mesh children
    // (ie defines something visible in the scene)
    // I've found some very unwindy FBX files with several thousand
    // nodes that served no practical purposes,
    // This also trims deformers and limb nodes, which we currently
    // do not support
    if mesh_leaf {
        let fbx_object = FbxObject {
            name,
            transform: data.as_local_transform(parent.as_ref().map(|p| p.global)),
            children: node.child_models().map(|c| c.object_id()).collect(),
        };
        hierarchy.insert(node.object_id(), fbx_object);
    }
    mesh_leaf
}
