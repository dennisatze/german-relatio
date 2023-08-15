# German relatio - for extracting narratives from German texts

Humans are story-tellers and narratives are an interesting subject to study for social scientists. [Ash et al. (2023)](https://www.cambridge.org/core/journals/political-analysis/article/relatio-text-semantics-capture-political-and-economic-narratives/E72C0482A44C9A817E381B394A73E2D6) demonstrate an empirical way of extracting core narratives from large text corpora. Read the full research paper in order to understand the underlying theoretical framework and to get a full description of the complete algorithm. (see this [link](https://arxiv.org/abs/2108.01720) for the arxiv version.) In essence, the idea is to see the extraction of narratives as a problem of dimension reduction. Using semantic role labeling, clustering and word embeddings, varities of the same core narratives are clustered together, which makes them useful for further text analysis tools. The algorithm is available from the [relatio](https://github.com/relatio-nlp/relatio) package. 

The code in this repository is a (preliminary) extension of their work by adjusting it to German texts. 

## Prerequisites

I recommend setting up a distinct python environment for tweaking with the code, e.g. with

```bash
python -m venv env
source ./env/bin/activate
```

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

This assumes you are on a Unix system (MacOS or Linux) and you are using pip and venv to set up the environment. Please consult the respective documentation if you want to [set up an environment on Windows](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or if you are using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 


## Specific adjustments

At the time of writing, I did not find a reliable pre-trained model for semantic role labeling of German texts, I therefore adjusted the original relatio algorithm to rely on syntactic parsing instead. This is done using spacy's syntactic parsing.



## Example usage

For illustrative purposes, I apply the relatio algorithm to a corpus of speeches from the Bundetag, the German parliament. You can find it in [example_usage.ipynb](./code/example_usage.ipynb). 
The corpus is a subset covering all speeches since 2015, if you are interest in this corpus and would like to have the full data, have a look at the [OpenDiscourse project](https://opendiscourse.de/).


## References

Ash, Elliott, Germain Gauthier, and Philine Widmer (2023). Text Semantics Capture Political and Economic Narratives. *Political Analysis* (First View): 1-18. https://doi.org/10.1017/pan.2023.8

OpenDiscourse Project. (2022). Plenarprotokolle des Deutschen Bundestags. https://opendiscourse.de/ <br>
See also: https://github.com/open-discourse/open-discourse
