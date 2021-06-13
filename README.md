# CNN-VAE-based-Trajectory-Classification-and-Anomaly-Detection
This is the code for the paper titled "Vehicular Trajectory Classification and Traffic Anomaly Detection in Videos Using a Hybrid CNN-VAE Architecture" https://arxiv.org/pdf/1812.07203.pdf


# Training

1) Train the CNN and VAE models for the respective scene and save the models

# Testing
1) Once the training is done use the test trajectories and covert them to gradient images and pass them to the CNN and VAE pipeline to obtain the results
2) Visualization can be done using the utility functions

# Real-time usage
Once the trained models have been saved the new trajectories obtained from the tracker can be gradient converted and be passed to the pipeline to decide about anomaly 
