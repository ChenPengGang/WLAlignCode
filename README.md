# WLAlign
Code for the paper "WL-Align: Weisfeiler-Lehman Relabeling for Aligning Users Across Networks via Regularized Representation Learning"

## Overview
The package contains the following folders and files:
- data: including the datasets used in the paper. Datasets are of the same format. Example illustrations are as below.
	- ACM-DBLP
	- Foursquare-Twitter
	- phone-email
	- MirrorNetwork:mirror network with perturbation from 0%-300%.
- network: 
	- network.py: used to record network data and the label changes
- model:
	- AggregateModel.py: the model used to do label aggregate.
	- EmbeddingModel.py: the model used to do Representation Learning
- utils: 
	- util.py :contains commonly used file manipulation functions
	- util4Agg.py :contains the functions required to get the label
	- util4Mapping.py :contains the functions required for label mapping
	- utils4ReadData.py :contains functions needed to read different datasets
- main.py: main code to run the whole algorithm
- testpAtN.py: test to compute the p@N.

## Prerequisites

python 3.7

torch >= 2.6.0

networkx == 2.6.3

## To run

Simply run the main.py. Change dataset and ratio per demand.

main.py --dataset ACM-DBLP --ratio 5

Simply run the testpAtN.py to test p@N of the result. Change dataset and ratio per demand.

testpAtN.py --dataset ACM-DBLP --ratio 5
