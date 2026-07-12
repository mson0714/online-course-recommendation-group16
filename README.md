# Online Course Recommendation - DS 423 Group 16

Content-based top-N recommender for the Coursera Courses Dataset 2021.

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
python recommendation_system.py --data Coursera_cleaned_for_recommendation.csv
```

Outputs are written to `outputs/`: metrics, charts, run metadata, and an example
top-10 list. The random query sample is reproducible (`random_state = 42`).

## Visual demo

Open `demo_recommender.html` directly in a modern browser. The page works
offline and lets the presenter choose a query course, switch between Top 5 and
Top 10, inspect recommendation scores, and explain the evaluation metrics.

The HTML page contains precomputed outputs from the verified Python model for a
smooth offline presentation. It does not refit TF-IDF in the browser; live model
fitting and evaluation are performed by `recommendation_system.py`.

## Submission files

- `recommendation_system.py`: runnable recommender and evaluation pipeline.
- `Coursera_cleaned_for_recommendation.csv`: cleaned course catalog used by the script.
- `demo_recommender.html`: standalone offline visual demonstration.
- `Online-Course-Recommendation.pptx`: project presentation.
- `Group16_Online_Course_Recommendation_Report_Professional_TranLanh&TrinhMinhSon.tex`:
  editable XeLaTeX report source.
- `Group16_Online_Course_Recommendation_Report_Professional_TranLanh&TrinhMinhSon.pdf`:
  final professional report.

## Use from Python

```python
from recommendation_system import load_courses, fit_model, recommend_by_title

courses = load_courses("Coursera_cleaned_for_recommendation.csv")
_, course_matrix = fit_model(courses)
print(recommend_by_title(courses, course_matrix, "Machine Learning", n=10))
```

## Limitations

Skill-overlap relevance favors courses with similar metadata and cannot measure
novelty, diversity, completion, or satisfaction. A production evaluation should
use anonymized learner interactions, temporal train/test splits, and online A/B
testing. The script intentionally avoids claiming collaborative filtering from
catalog-only data.

## Data source

Kapoor, K. (2021). *Coursera Courses Dataset 2021* [Data set]. Kaggle.
https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021
