use bevy::prelude::{AddAsset, App, Plugin, Resource, Material, StandardMaterial};

pub use data::{FbxMesh, FbxScene};
pub use loader::FbxLoader;

pub(crate) mod data;
pub(crate) mod fbx_transform;
pub(crate) mod loader;
pub mod material_loader;
pub(crate) mod utils;

use material_loader::MaterialLoader;

/// Adds support for FBX file loading to the app.
#[derive(Default)]
pub struct FbxPlugin;

/// Resource to control which material loaders the `FbxLoader`
/// uses.
///
/// See [`MaterialLoader`] documentation for more details.
///
/// You can define your own by inserting this as a resource
/// **before** adding the `FbxPlugin` to the app.
/// If you define your own, make sure to add back the default
/// fallback methods if you need them!
///
/// The default loaders are defined by [`material_loader::default_loader_order`].
#[derive(Clone, Resource)]
pub struct FbxMaterialLoaders<M: Material>(pub Vec<MaterialLoader<M>>);
impl Default for FbxMaterialLoaders<StandardMaterial> {
    fn default() -> Self {
        Self(material_loader::default_loader_order().into())
    }
}

impl Plugin for FbxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<FbxLoader<StandardMaterial>>()
            .add_asset::<FbxMesh<StandardMaterial>>()
            .add_asset::<FbxScene<StandardMaterial>>();
    }
}
