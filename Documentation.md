# Definitions
* Resource: A file located in either the game's install directory or in its asset package file.
    * Script: A piece of code that has access to whatever it is bound to. Scripts are used by the Director, SubDirector, Sprites, and Actors.
    * Model: A 3D model, used by Props and Actors. Vega3D uses .vmdl files, which contain vertexes, normals, UVs, meshes, and bones.
    * Texture: A texture, used by Materials. Many image formats are supported, such as PNG, JPG, BMP, and binary representations.
    * Material: A material, used by Lights, Props and Actors. Vega3D has .vmtl files, which contain references to basic, specular, and diffuse textures.
    * Animation: An animation, either of a skeleton, transformation, model, or texture. Vega3D uses .vani files.
    * Music/SFX: Music or sound effects that may be used in gameplay. 
    * Plan: A list of resources to be loaded into memory at certain Cues. Can be used to initialize scenes.
    * Shader: A shader that may be used in a Pipeline.
    * Other: Any file that may be of use, such as subtitles, keybinds, or save files. The contents of these files can be read freely.
* Objects: Any object used during gameplay.
    * Director and SubDirector: A Script which has control over the entire scene. Directors are always active, while SubDirectors control a single scene.
    * Scene: A collection of objects that are loaded in on a Cue. 
    * Stage: An unmoving collection of Props and Actors that make up the "world".
    * Prop: A stationary object that has no scripting, but may animate. Usually, you want Props for background elements, terrain, etc.
    * Actor: A dynamic object with scripting and animation. For example, the player, items, and enemies are Actors.
    * ActorGroup: A group of actors, and can be used to classify Actors into categories.
    * Sprite: Similar to an Actor, but in 2D. Can be used for UI elements, text, etc.
    * Light: A light source, such as the Sun or a lightbulb.
    * Cue: A global, subscribable event, which can be used to implement loading screens, cutscenes, etc.
    * Camera: A camera, which can be targeted as the main renderer or render to a texture.
