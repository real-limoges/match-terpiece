###   
"If I have seen further, it is by standing on the shoulders of giants."  

-- Sir Isaac Newton

### What is Transfer Learning?

Transfer learning takes a machine learning model that solves one classification problem and recycles the effort spent to solve other related problems. In the case of this project, the neural network was trained on ImageNet photos. ImageNet images are pictures of nature; different species of birds, cats, dogs, and other animals are the objects of classifcation. This project is interested in clustering art pieces - few of which have any relation to ImageNet.

### Which Pre-Trained Neural Network Was Used?

VGG16, also known as OxfordNet.

VGG16 was the 2014 ImageNet winner; while there have been recent advances in neural networks, it still performs excellently on its intended purpose and prediction is relatively quick.

VGG16, as many past winners of the ImageNet competition, uses convolutional layers to process the data. Convolutional layers serve two purposes in helping classify images.  

*   Convolutions help decrease the number of neurons in a network. ImageNet items are 224x224 pixels in 3 color channels. For a fully connected neural network, this would require ~150k parameters for each layer. We would not be able to leverage the full capacity of deep learning with such shallow layers.
*   Convolutions help maintain the structure of the image that is passed through it.


### What Data Was Used

I used the Flickr API to generate the links of approximately 55 thousand images that were tagged as "abstract art." Of these, 45 thousand images were publically available and satisfied the 224x224 pixel requirement.

Images

### Is There Pattern In The Data?

Absolutely! The activations in the first fully connected layer can be used as the data for any number of clustering algorithms. The dense representaiton of the images clusters quite naturally. For some examples, I have a couple in the [Gallery](www.match-terpiece.com/gallery) on my website.
