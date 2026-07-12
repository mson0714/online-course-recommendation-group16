# Online Course Recommendation - DS 423 Group 16

Content-based top-N recommender for the Coursera Courses Dataset 2021.

Repository: https://github.com/mson0714/online-course-recommendation-group16

The submission also includes the professional project report in both LaTeX
source and compiled PDF form, plus the PowerPoint presentation.

## Team

- Trần Lãnh - 28211151726 (Data & Analysis Lead, 50%)
- Trịnh Minh Son - 28211144373 (Modeling & Evaluation Lead, 50%)

## Method

The model represents every course with TF-IDF features from its title, skills,
university, difficulty and description. Cosine similarity retrieves related
courses, and normalized rating provides a 5% tie-breaker. Because the supplied
catalog has no learner-course interaction records, collaborative filtering is
not valid for this dataset. Offline Precision@K and Recall@K use a declared
proxy: another course is relevant when skill-set Jaccard similarity is at least
0.20. This evaluates catalog retrieval, not real user satisfaction.

## Run

```bash
python -m pip install -r requirements.txt
jupyter notebook Group16_Online_Course_Recommendation.ipynb
```

In Jupyter, select **Cell → Run All** or execute the cells in order. The notebook
is fully self-contained and reads `Coursera_cleaned_for_recommendation.csv`
from the same directory. The random evaluation sample is reproducible
(`random_state = 42`).

## Visual demo

Open `demo_recommender.html` directly in a modern browser. The page works
offline and lets the presenter choose a query course, switch between Top 5 and
Top 10, inspect recommendation scores, and explain the evaluation metrics.

The HTML page contains precomputed outputs from the verified notebook model for a
smooth offline presentation. It does not refit TF-IDF in the browser; live model
fitting and evaluation are performed by the notebook.

## Submission files

- `Group16_Online_Course_Recommendation.ipynb`: complete runnable recommender,
  organized into data, TF-IDF, recommendation, evaluation, charts, and Q&A sections.
- `Coursera_cleaned_for_recommendation.csv`: cleaned course catalog used by the notebook.
- `demo_recommender.html`: standalone offline visual demonstration.
- `Online-Course-Recommendation.pptx`: project presentation.
- `Group16_Online_Course_Recommendation_Report_Professional_TranLanh&TrinhMinhSon.tex`:
  editable XeLaTeX report source.
- `Group16_Online_Course_Recommendation_Report_Professional_TranLanh&TrinhMinhSon.pdf`:
  final professional report.

## Limitations

Skill-overlap relevance favors courses with similar metadata and cannot measure
novelty, diversity, completion, or satisfaction. A production evaluation should
use anonymized learner interactions, temporal train/test splits, and online A/B
testing. The notebook intentionally avoids claiming collaborative filtering
from catalog-only data.

## Data source

Kapoor, K. (2021). *Coursera Courses Dataset 2021* [Data set]. Kaggle.
https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021
