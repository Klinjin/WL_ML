# Starting Kit and Sample Submission
***



## Starting Kit
We have prepared a starting kit to help participants get started with the competition, to understand the data and prepare submissions for Codabench. You can check the starting kit notebook on our GitHub repository
### [Starting Kit Notebook](https://github.com/FAIR-Universe/Cosmology_Challenge/blob/master/Startingkit_WL_PSAnalysis.ipynb)

#### ⚠️ Note:
- To run the starting kit, you will need the public/training data. You can download it from the `Files` tab or from [Here](https://www.codabench.org/datasets/download/c99c803a-450a-4e51-b5dc-133686258428/)



## Dummy Sample Submission
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

### ⬇️ [Dummy Sample Submission](https://www.codabench.org/datasets/download/65bc826a-a635-4fe5-a20e-89efa8533ad8/)
