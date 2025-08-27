# Starting Kit and Sample Submission
***


## Phase 1: Cosmological Parameter Estimation
### Starting Kit
We have prepared a starting kit to help participants get started with the competition, to understand the data and prepare submissions for Codabench. You can check the starting kit notebook on our GitHub repository or through the Google Colab
### [<ins>Starting Kit Notebook</ins>](https://github.com/FAIR-Universe/Cosmology_Challenge/blob/master/Phase_1_Startingkit_WL_PSAnalysis.ipynb)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iySb87VmyCtz6y8Zg367xR6hetD6gKTi?usp=sharing)


#### ⚠️ Note:
- To run the starting kit, you will need the public/training data. You can download it from the `Files` tab or from [<ins>Here</ins>](https://www.codabench.org/datasets/download/c99c803a-450a-4e51-b5dc-133686258428/).



### Dummy Sample Submission
Dummy sample submission is provided to make you understand what is expected as a submission. The sample submission is a zip that only contains one json file named `result.json`. This file contains lists of `means` and `errorbars`. Each list has `NTest` total number of items and  each item of these lists contains 2 values each. The format looks like this:

```json
{
    "means": [
        [
            2.1234,
            3.1456
        ],
        ... # total 4000 items
    ],
    "errorbars": [
        [
            0.1234,
            0.1456
        ],
        ... # total 4000 items
    ]
}
```

### ⬇️ [<ins>Dummy Sample Submission</ins>](https://www.codabench.org/datasets/download/65bc826a-a635-4fe5-a20e-89efa8533ad8/)


## Phase 2: Out-of-Distribution Detection
The starting kit of Phase 2 will be available when the Phase 2 starts.