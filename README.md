# WLAlign
Code for the paper "Weisfeiler-Lehman guided Representation Learning for User Alignment across Social Networks"

## Overview
The package contains the following folders and files:
- data: including the datasets used in the paper. Datasets are of the same format. Example illustrations are as below.
	- ACM-DBLP
	- Foursquare-Twitter
	- phone-email
	- MirrorNetwork:mirror network with perturbation from 0%-450%.
- network: 
	- network.py: used to record network data and the label changes
- model:
	- AggregateModel.py: the model used to do label aggregate.
	- EmbeddingModel.py: the model used to do Representation Learning
- utils: 
	- util.py
	- util4Agg.py
	- util4Mapping.py
	- utils4ReadData.py
- main.py: main code to run the whole algorithm
- testqAtN.py: test to compute the p@N.

## Prerequisites

python 3.7
torch >= 2.6.0
networkx == 2.6.3

## To run

Simply run the main.py. Change dataset and ratio per demand.

main.py --dataset ACM-DBLP --ratio 5

Simply run the testqAtN.py to test p@N of the result. Change dataset and ratio per demand.

testqAtN.py --dataset ACM-DBLP --ratio 5