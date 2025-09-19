# Visualizing Herbarium Plant Specimens 
This task utilizes a SWIN Transformer fine-tuned on the top 50 species form the Asteraceae family, based on the FGVC9 2022 dataset. After extracting feature vectors from the fine-tuned model, dimensionality reduction techniques were applied to visualize the learned representations.

Two methods were explored: PCA and t-SNE. t-SNE demonstrated clearer groupings and visual separations of species than PCA. 

Experimented with:
1. Extracting feature vectors from SWIN Finetuned on top 50 species from Asteraceae family and then applying dimensionality reduction techniques.
2. Extracting feature vecotrs from SWIN Finetuned on all species from FGVC 2022 dataset and then applying dimensionality reduction techniques.


## Overview
1. `kaggle22_clustering.ipynb` - notebook to get feature vectors using finetuned SWIN Transformer and then performing dimensionality reduction using PCA and t-SNE 
2. `index.html` - to view visualization using t-SNE for Top 50 most occurring species in the Asteraceae Family. Points on the graph are clickable, view the herbarium specimen associated with that specific point and compare up to two images at a time. Change the JSON file to visualize a different plot. 

## Running the Visualization 
You will first need to be added to the `herbdl` project on the SCC. 

You can view the t-SNE visualization here:
`/projectnb/herbdl/workspaces/mvoong/herbdl/clustering_viz/index.html`

In the plot, each point represents a herbarium specimen. You can click on any point to view the associated image. All images are stored within the `/projectnb/herbdl` project, which is how the visualization is able to access and display them. The image display operates like a stack with the ability to display up to two images. 

`/projectnb/herbdl/workspaces/mvoong/herbdl/clustering_viz/asteraceae_tsne_plot_astera_50checkpoint-1300.json`
- SWIN Finetuned on top 50 species from Asteraceae family. 

`/projectnb/herbdl/workspaces/mvoong/herbdl/clustering_viz/asteraceae_tsne_plot_checkpoint-139125.json`
- SWIN Finetuned on all species from FGVC 2022 dataset. 