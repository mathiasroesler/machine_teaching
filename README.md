# Machine teaching for mutli-class image classification
This project uses neural networks for image classification to test the 
performances of machine teaching. They were tested using two different databases
(MNIST and CIFAR-10), and two architectures (variants of LeNet-5 [1] and AllCNN [2])

## General information
The objective of this project is to apply machine teaching to image classification using well-known databases. The machine teaching algorithm was inspired by the work of Dagustupa et al. [3]. The performances of machine teaching to train neural networks were compared to other similar strategies, namely curriculum learning and self-paced learning. This work was conducted during a 5-month internship at the ISIR laboratory in Paris, France to complete my Master's degree in robotics and artificial intelligence.

## Technologies
* Python 3.8.2
* Tensorflow 2.2.0
* Numpy 1.18.4

## Launch
To launch the project from the code folder use the command `python3 main.py database file_name`

The database is either mnist or cifar. The file is the location at which to save the results.

To view the results use the command `python3 plot_fct.py file_name database`
	
The file is the location at which the results were saved. The database is either cifar or mnist and defaults to mnist if an erroneous name was inputed.

## Results
Some results have been added using different values of M. This variable limits the number of examples in the teaching set to M% of the training set.
The results contain graphs of the training accuracy, the test accuracy and the training time for the two architectures and the two databases.

## References
[1] Y. Lecun, “Gradient-based learning applied to document recognition". PROCEEDINGS OF THE IEEE, vol 86, no. 11, p.47, 1998.

[2] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller, “Striving for simplicity: The all convolutional net”,arXiv:1412.6806 [cs], Apr. 13, 2015. arXiv:1412.6806.[Online]. Available:http://arxiv.org/abs/1412.6806

[3] S. Dasgupta, D. Hsu, S. Poulis, and X. Zhu, “Teaching a black-box learner”, inProceedings of the 36th International Conference on Machine Learning, K. Chaud-huri and R. Salakhutdinov, Eds., ser. Proceedings of Machine Learning Research,vol. 97, Long Beach, California, USA: PMLR, Sep. 2019, pp. 1547–1555. [Online] Available:http://proceedings.mlr.press/v97/dasgupta19a.html
