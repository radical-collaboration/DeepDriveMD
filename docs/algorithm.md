* **Simulation Tasks**: Each simulation uses `OpenMM` and executes on a single GPU
* **Task**: Collect trajectory information
* **Task**: Create contact map of frames in .h5 format → this task generates the input to CVAE model
* **CVAE/autoencoder Tasks**: Each CVAE model uses the `CUDA` backend and runs on a single GPU, CVAE's differ in hyper_dims 
* **Task**: Collect weights
* **Task**: While True loop (for n_iterations):
  * Check # of frames
  * Perform inference for each model latent dimension (Check if this needs a GPU) → collect outliers
   * Writes new pdb traj files
  * Determine new outliers
  * Calculate RMSD
  * If conditions are met (len(traj) <10K and outliers present, and if GPU available): 
  * **Tasks**: (Simulations for outliers) 
  * Increase the iteration counter
