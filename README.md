# Machine Learning
Repository Description:
Different machine learning algorithms implemented from scratch.

Note:
The project_Bagged_Linear_SVM.py is a project I did for school. 
The project involved a simulated dataset of single nucleotide polymorphism (SNP) genotype data 
containing 29623 SNPs (total features). Amongst all SNPs are 15 causal 
ones which means they and neighboring ones discriminate between case and 
controls while remainder are noise. In this training data there were 4000 cases and 4000 controls.
I was also provided training labels for the training data. 
The task was to predict the labels of 2000 test individuals whose true labels are known to the professor. 
The goal is to achieve an accuracy of at least 63% and to get full point.
So I implemented 100 bags of linear SVM with a C of 0.1 from sklearn. I first did a 70-30 split 
Then, I featured selected via filter method using chi-square and selected the top 12 features. 
On the 30% test data I was getting an accuracy of 66%.
