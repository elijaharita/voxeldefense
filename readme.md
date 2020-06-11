# v1

## Primary Objective
Create a simple, limited interactable voxel world

## Features / specs
- 0.25 meter voxels
- 32^3 voxels per chunk
- 8^3 chunks in world
- randomly generated world
- placing and destroying at least 1 type of block
- each render object is 1 chunk
- chunks raytraced on compute shader
- physics optional

## Render Instance Responsibilities
- clear screen
- allocate chunk
- deallocate chunk
- set chunk contents
- draw chunk at position
- set camera information
- set viewport information

## Render Instance Organization
- Gpu
  - handle instance creation and physical device management
- Render Instance
  - takes a physical device
  - contain the pipeline and scene rendering information
  - expose functions for the render instance responsibilities