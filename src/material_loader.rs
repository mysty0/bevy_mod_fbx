use std::path::Path;

#[cfg(feature = "maya_3dsmax_pbr")]
use crate::utils::fbx_extend::*;

use anyhow::Context;
use bevy::{
    pbr::{AlphaMode, StandardMaterial},
    prelude::{Color, Handle, Image, Material, debug},
    utils::{HashMap, BoxedFuture}, asset::{LoadContext, LoadedAsset}, render::{texture::{CompressedImageFormats, ImageSampler, ImageType}, render_resource::{AddressMode, SamplerDescriptor}},
};
use fbxcel_dom::v7400::{data::{material::ShadingModel, texture::WrapMode}, object::{material::MaterialHandle, self, texture::TextureHandle}};
use rgb::RGB;

pub struct TextureLoader<'a, 'w> {
    pub textures: &'a mut HashMap<String, Handle<Image>>,
    pub load_context: &'a mut LoadContext<'w>,
    pub suported_compressed_formats: CompressedImageFormats,
}

impl<'a, 'w> TextureLoader<'a, 'w> {
    pub async fn get_cached_texture(
        &mut self,
        texture_handle: object::texture::TextureHandle<'_>,
    ) -> anyhow::Result<Handle<Image>> {
        let handle_label = match texture_handle.name() {
            Some(name) if !name.is_empty() => format!("FbxTexture@{name}"),
            _ => format!("FbxTexture{}", texture_handle.object_id().raw()),
        };

        // Either copy the already-created handle or create a new asset
        // for each image or texture to load.
        if let Some(handle) = self.textures.get(&handle_label) {
            debug!("Already encountered texture: {handle_label}, skipping");
            Ok(handle.clone())
        } else {
            let texture = self.get_texture(texture_handle).await?;
            let handle = self
                .load_context
                .set_labeled_asset(&handle_label, LoadedAsset::new(texture));
            self.textures.insert(handle_label, handle.clone());
            Ok(handle)
        }
    }

    pub async fn get_texture(
        &self,
        texture_obj: object::texture::TextureHandle<'_>,
    ) -> anyhow::Result<Image> {
        let properties = texture_obj.properties();
        let address_mode_u = {
            let val = properties
                .wrap_mode_u_or_default()
                .context("Failed to load wrap mode for U axis")?;
            match val {
                WrapMode::Repeat => AddressMode::Repeat,
                WrapMode::Clamp => AddressMode::ClampToEdge,
            }
        };
        let address_mode_v = {
            let val = properties
                .wrap_mode_v_or_default()
                .context("Failed to load wrap mode for V axis")?;
            match val {
                WrapMode::Repeat => AddressMode::Repeat,
                WrapMode::Clamp => AddressMode::ClampToEdge,
            }
        };
        let video_clip_obj = texture_obj
            .video_clip()
            .context("No image data for texture object")?;

        let image: Result<Image, anyhow::Error> = self.load_video_clip(video_clip_obj).await;
        let mut image = image.context("Failed to load texture image")?;

        image.sampler_descriptor = ImageSampler::Descriptor(SamplerDescriptor {
            address_mode_u,
            address_mode_v,
            ..Default::default()
        });
        Ok(image)
    }

    pub async fn load_video_clip(
        &self,
        video_clip_obj: object::video::ClipHandle<'_>,
    ) -> anyhow::Result<Image> {
        debug!("Loading texture image: {:?}", video_clip_obj.name());

        let relative_filename = video_clip_obj
            .relative_filename()
            .context("Failed to get relative filename of texture image")?;
        debug!("Relative filename: {:?}", relative_filename);

        let file_ext = Path::new(&relative_filename)
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_ascii_lowercase();
        let image: Vec<u8> = if let Some(content) = video_clip_obj.content() {
            // TODO: the clone here is absolutely unnecessary, but there
            // is no way to reconciliate its lifetime with the other branch of
            // this if/else
            content.to_vec()
        } else {
            let parent = self.load_context.path().parent().unwrap();
            let clean_relative_filename = relative_filename.replace('\\', "/");
            let image_path = parent.join(clean_relative_filename);
            self.load_context.read_asset_bytes(image_path).await?
        };
        let is_srgb = false; // TODO
        let image = Image::from_buffer(
            &image,
            ImageType::Extension(&file_ext),
            self.suported_compressed_formats,
            is_srgb,
        );
        let image = image.context("Failed to read image buffer data")?;
        debug!(
            "Successfully loaded texture image: {:?}",
            video_clip_obj.name()
        );

        Ok(image)
    }
}

pub trait RawMaterialLoader<M: Material>: Sync {
    fn load<'a, 'w>(
        &'a self, 
        texture_loader: &'a mut TextureLoader<'a, 'w>, 
        material_obj: object::material::MaterialHandle<'a>
    ) -> BoxedFuture<'a, anyhow::Result<Option<M>>>;
}

impl <F, M: Material> RawMaterialLoader<M> for F
where
  F: for<'a, 'w> Fn(&'a mut TextureLoader<'a, 'w>, object::material::MaterialHandle<'a>) -> BoxedFuture<'a, anyhow::Result<Option<M>>> + Sync
{
    fn load<'a, 'w>(
        &'a self, 
        texture_loader: &'a mut TextureLoader<'a, 'w>, 
        material_obj: object::material::MaterialHandle<'a>
    ) -> BoxedFuture<'a, anyhow::Result<Option<M>>> {
        self(texture_loader, material_obj)
    }
}

impl<M: Material> RawMaterialLoader<M> for MaterialLoader<M> {
    fn load<'a, 'w>(
        &'a self,
        texture_loader: &'a mut TextureLoader<'a, 'w>, 
        material_obj: object::material::MaterialHandle<'a>
    ) -> BoxedFuture<'a, anyhow::Result<Option<M>>> {
        Box::pin(async move {
            use crate::utils::fbx_extend::*;
            enum TextureSource<'a> {
                Processed(Image),
                Handle(TextureHandle<'a>),
            }
            let mut textures = HashMap::default();
            // code is a bit tricky so here is a rundown:
            // 1. Load all textures that are meant to be preprocessed by the
            //    MaterialLoader
            for &label in self.dynamic_load {
                if let Some(texture) = material_obj.find_texture(label) {
                    let texture = texture_loader.get_texture(texture).await?;
                    textures.insert(label, texture);
                }
            }

            (self.preprocess_textures)(material_obj, &mut textures);
            // 2. Put the loaded images and the non-preprocessed texture labels into an iterator
            let mut texture_handles = HashMap::with_capacity(textures.len() + self.static_load.len());
            let texture_handles_iter = textures
                .drain()
                .map(|(label, image)| (label, TextureSource::Processed(image)))
                .chain(self.static_load.iter().filter_map(|l| {
                    material_obj
                        .find_texture(l)
                        .map(|te| (*l, TextureSource::Handle(te)))
                }));
            // 3. For each of those, create an image handle (with potential caching based on the texture name)
            for (label, texture) in texture_handles_iter {
                let handle_label = match texture {
                    TextureSource::Handle(texture_handle) => match texture_handle.name() {
                        Some(name) if !name.is_empty() => format!("FbxTexture@{name}"),
                        _ => format!("FbxTexture{}", texture_handle.object_id().raw()),
                    },
                    TextureSource::Processed(_) => match material_obj.name() {
                        Some(name) if !name.is_empty() => format!("FbxTextureMat@{name}/{label}"),
                        _ => format!("FbxTextureMat{}/{label}", material_obj.object_id().raw()),
                    },
                };

                // Either copy the already-created handle or create a new asset
                // for each image or texture to load.
                let handle = if let Some(handle) = texture_loader.textures.get(&handle_label) {
                    debug!("Already encountered texture: {label}, skipping");

                    handle.clone()
                } else {
                    let texture = match texture {
                        TextureSource::Processed(texture) => texture,
                        TextureSource::Handle(texture) => texture_loader.get_texture(texture).await?,
                    };
                    let handle = texture_loader
                        .load_context
                        .set_labeled_asset(&handle_label, LoadedAsset::new(texture));
                    texture_loader.textures.insert(handle_label, handle.clone());
                    handle
                };
                texture_handles.insert(label, handle);
            }
            // 4. Call with all the texture handles
            Ok((self.with_textures)(material_obj, texture_handles))
        })
    }
}

/// Load materials from an FBX file.
///
/// Define your own to extend `bevy_mod_fbx`'s material loading capabilities.
#[derive(Clone, Copy)]
pub struct MaterialLoader<M: Material> {
    /// The FBX texture field name used by the material you are loading.
    ///
    /// Textures declared here are directly passed to `with_textures` without modification,
    /// this enables caching and re-using textures without re-reading the files
    /// multiple times over.
    ///
    /// They are loaded by the [`FbxLoader`] and provided to the other functions
    /// defined in the rest of this struct, associated with their names in a `HashMap`.
    ///
    /// [`FbxLoader`]: crate::FbxLoader
    pub static_load: &'static [&'static str],

    /// The FBX texture field name used by textures you wish to transform.
    ///
    /// Textures declared here are passed to `preprocess_textures` for further
    /// processing, enabling preprocessing.
    ///
    /// They are loaded by the [`FbxLoader`] and provided to the other functions
    /// defined in the rest of this struct, associated with their names in a `HashMap`.
    ///
    /// [`FbxLoader`]: crate::FbxLoader
    pub dynamic_load: &'static [&'static str],

    /// Run some math on the loaded textures, handy if you have to convert between texture
    /// formats or swap color channels.
    ///
    /// To update, remove or add textures, return the `HashMap` with the new values.
    ///
    /// The `Image`s are then added to the asset store (`Assets<Image>`) and a handle
    /// to them is passed to `with_textures` in additions to the handles of the textures
    /// declared in the `static_load` field.
    pub preprocess_textures: fn(MaterialHandle, &mut HashMap<&'static str, Image>),

    /// Create and return the bevy [`StandardMaterial`] based on the [`Handle<Image>`] loaded
    /// from the return value of `preprocess_textures`.
    pub with_textures:
        fn(MaterialHandle, HashMap<&'static str, Handle<Image>>) -> Option<M>,
}

const SPECULAR_TO_METALLIC_RATIO: f32 = 0.8;

/// Load Lambert/Phong materials, making minimal effort to convert them
/// into bevy's PBR material.
///
/// Note that the conversion has very poor fidelity, since Phong doesn't map well
/// to PBR.
pub const LOAD_LAMBERT_PHONG: MaterialLoader<StandardMaterial> = MaterialLoader {
    static_load: &[
        "NormalMap",
        "EmissiveColor",
        "DiffuseColor",
        "TransparentColor",
    ],
    dynamic_load: &[],
    preprocess_textures: |_, _| {},
    with_textures: |material_obj, textures| {        
        use AlphaMode::{Blend, Opaque};
        use ShadingModel::{Lambert, Phong};
        let properties = material_obj.properties();
        let shading_model = properties
            .shading_model_or_default()
            .unwrap_or(ShadingModel::Unknown);
        if !matches!(shading_model, Lambert | Phong) {
            return None;
        };
        let transparent = textures.get("TransparentColor").cloned();
        let is_transparent = transparent.is_some();
        let diffuse = transparent.or_else(|| textures.get("DiffuseColor").cloned());
        let base_color = properties
            .diffuse_color_or_default()
            .map_or(Default::default(), ColorAdapter)
            .into();
        let specular = properties.specular_or_default().unwrap_or_default();
        let metallic = (specular.r + specular.g + specular.b) / 3.0;
        let metallic = metallic as f32 * SPECULAR_TO_METALLIC_RATIO;
        let roughness = properties
            .shininess()
            .ok()
            .flatten()
            .map_or(0.8, |s| (2.0 / (2.0 + s)).sqrt());
        Some(StandardMaterial {
            alpha_mode: if is_transparent { Blend } else { Opaque },
            // For bistro only
            // alpha_mode: AlphaMode::Mask(0.8),
            base_color,
            metallic,
            perceptual_roughness: roughness as f32,
            emissive_texture: textures.get("EmissiveColor").cloned(),
            base_color_texture: diffuse,
            normal_map_texture: textures.get("NormalMap").cloned(),
            flip_normal_map_y: true,
            ..Default::default()
        })
    },
};

/// The default material if all else fails.
///
/// Picks up the non-texture material values if possible,
/// otherwise it will just look like white clay.
pub const LOAD_FALLBACK: MaterialLoader<StandardMaterial> = MaterialLoader {
    static_load: &[],
    dynamic_load: &[],
    preprocess_textures: |_, _| {},
    with_textures: |material_obj, _| {
        let properties = material_obj.properties();
        let base_color = properties
            .diffuse_color()
            .ok()
            .flatten()
            .map(|c| ColorAdapter(c).into())
            .unwrap_or(Color::PINK);
        let metallic = properties
            .specular()
            .ok()
            .flatten()
            .map(|specular| (specular.r + specular.g + specular.b) / 3.0)
            .map(|metallic| metallic as f32 * SPECULAR_TO_METALLIC_RATIO)
            .unwrap_or(0.2);
        let roughness = properties
            .shininess()
            .ok()
            .flatten()
            .map_or(0.8, |s| (2.0 / (2.0 + s)).sqrt());
        Some(StandardMaterial {
            base_color,
            perceptual_roughness: roughness as f32,
            alpha_mode: AlphaMode::Opaque,
            metallic,
            ..Default::default()
        })
    },
};

#[cfg(feature = "maya_3dsmax_pbr")]
mod maya_consts {
    pub const PBR_TYPE_ID: i32 = 1166017;
    pub const DEFAULT_ROUGHNESS: f32 = 0.089;
    pub const DEFAULT_METALIC: f32 = 0.01;
}

// Note that it's impossible to enable the `maya_pbr` feature right now.
/// Load Maya's PBR material FBX extension.
///
/// This doesn't preserve environment maps or fresnel LUT,
/// since bevy's PBR currently doesn't support environment maps.
///
/// This loader is only available if the `maya_pbr` feature is enabled.
#[cfg(feature = "maya_3dsmax_pbr")]
pub const LOAD_MAYA_PBR: MaterialLoader<StandardMaterial> = MaterialLoader {
    static_load: &[
        "Maya|TEX_normal_map",
        "Maya|TEX_color_map",
        "Maya|TEX_ao_map",
        "Maya|TEX_emissive_map",
    ],
    dynamic_load: &["Maya|TEX_metallic_map", "Maya|TEX_roughness_map"],
    // FIXME: this assumes both metallic map and roughness map
    // are encoded in texture formats that can be stored as
    // a byte array in CPU memory.
    // This is not the case for compressed formats such as KTX or DDS
    // FIXME: this also assumes the texture channels are 8 bit.
    preprocess_textures: |material_handle, images| {
        use bevy::render::render_resource::{TextureDimension::D2, TextureFormat::Rgba8UnormSrgb};
        let mut run = || {
            // return early if we detect this material is not Maya's PBR material
            let mat_maya_type = material_handle.get_i32("Maya|TypeId")?;
            if mat_maya_type != maya_consts::PBR_TYPE_ID {
                return None;
            }
            let combine_colors = |colors: &[u8]| match colors {
                // Only one channel is necessary for the metallic and roughness
                // maps. If we assume the texture is greyscale, we can take any
                // channel (R, G, B) and assume it's approximately the value we want.
                &[bw, ..] => bw,
                _ => unreachable!("A texture must at least have a single channel"),
            };
            // Merge the metallic and roughness map textures into one,
            // following the GlTF standard for PBR textures.
            // The resulting texture should have:
            // - Green channel set to roughness
            // - Blue channel set to metallic
            let metallic = images.remove("Maya|TEX_metallic_map")?;
            let rough = images.remove("Maya|TEX_roughness_map")?;
            let image_size = metallic.texture_descriptor.size;
            let metallic_components =
                metallic.texture_descriptor.format.describe().components as usize;
            let rough_components = rough.texture_descriptor.format.describe().components as usize;
            let metallic_rough: Vec<_> = metallic
                .data
                .chunks(metallic_components)
                .zip(rough.data.chunks(rough_components))
                .flat_map(|(metallic, rough)| {
                    [0, combine_colors(rough), combine_colors(metallic), 255]
                })
                .collect();
            let metallic_rough = Image::new(image_size, D2, metallic_rough, Rgba8UnormSrgb);
            images.insert("Metallic_Roughness", metallic_rough);
            Some(())
        };
        run();
    },
    with_textures: |handle, textures| {
        // return early if we detect this material is not Maya's PBR material
        let mat_maya_type = handle.get_i32("Maya|TypeId");
        if mat_maya_type != Some(maya_consts::PBR_TYPE_ID) {
            return None;
        }
        let lerp = |from, to, stride| from + (to - from) * stride;
        // Maya has fields that tells how much of the texture should be
        // used in the final computation of the value vs the baseline value.
        // We set the `metallic` and `perceptual_roughness` to
        // lerp(baseline_value, fully_texture_value, use_map)
        // so if `use_map` is 1.0, only the texture pixel counts,
        // while if it is 0.0, only the baseline count, and anything inbetween
        // is a mix of the two.
        let has_rm_texture = textures.contains_key("Metallic_Roughness");
        let use_texture = if has_rm_texture { 1.0 } else { 0.0 };
        let use_metallic = handle
            .get_f32("Maya|use_metallic_map")
            .unwrap_or(use_texture);
        let use_roughness = handle
            .get_f32("Maya|use_roughness_map")
            .unwrap_or(use_texture);
        let metallic = handle
            .get_f32("Maya|metallic")
            .unwrap_or(maya_consts::DEFAULT_METALIC);
        let roughness = handle
            .get_f32("Maya|roughness")
            .unwrap_or(maya_consts::DEFAULT_ROUGHNESS);
        Some(StandardMaterial {
            flip_normal_map_y: true,
            base_color_texture: textures.get("Maya|TEX_color_map").cloned(),
            normal_map_texture: textures.get("Maya|TEX_normal_map").cloned(),
            metallic_roughness_texture: textures.get("Metallic_Roughness").cloned(),
            metallic: lerp(metallic, 1.0, use_metallic),
            perceptual_roughness: lerp(roughness, 1.0, use_roughness),
            occlusion_texture: textures.get("Maya|TEX_ao_map").cloned(),
            emissive_texture: textures.get("Maya|TEX_emissive_map").cloned(),
            alpha_mode: AlphaMode::Opaque,
            ..Default::default()
        })
    },
};

/// The default fbx material loaders.
///
/// If you don't provide your own in the [`FbxMaterialLoaders`] resource,
/// the ones declared in this will be used instead.
///
/// You can also use thise function if you want to add your own loaders
/// and still want to fallback to the default ones.
///
/// [`FbxMaterialLoaders`]: crate::FbxMaterialLoaders
pub const fn default_loader_order() -> &'static [MaterialLoader<StandardMaterial>] {
    &[
        #[cfg(feature = "maya_3dsmax_pbr")]
        LOAD_MAYA_PBR,
        LOAD_LAMBERT_PHONG,
        LOAD_FALLBACK,
    ]
}

#[derive(Default)]
struct ColorAdapter(RGB<f64>);
impl From<ColorAdapter> for Color {
    fn from(ColorAdapter(rgb): ColorAdapter) -> Self {
        Color::rgb(rgb.r as f32, rgb.g as f32, rgb.b as f32)
    }
}
