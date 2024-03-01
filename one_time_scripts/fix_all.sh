
echo "JUDICIAL"
mkdir /homes/etayl/code/bert/judicial
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/judicial/*" /homes/etayl/code/bert/judicial
python fix_data.py judicial 
vast5 sync "/homes/etayl/code/bert/judicial/*" s3://mobileye-team-angie/users/etay/nltk_data/judicial/
rm -rf /homes/etayl/code/bert/judicial

echo "MEDICAL"
mkdir /homes/etayl/code/bert/medical
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/medical/*" /homes/etayl/code/bert/medical
python fix_data.py medical 
vast5 sync "/homes/etayl/code/bert/medical/*" s3://mobileye-team-angie/users/etay/nltk_data/medical/
rm -rf /homes/etayl/code/bert/medical


echo "NEWS"
mkdir /homes/etayl/code/bert/news
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/news/*" /homes/etayl/code/bert/news
python fix_data.py news 
vast5 sync "/homes/etayl/code/bert/news/*" s3://mobileye-team-angie/users/etay/nltk_data/news/
rm -rf /homes/etayl/code/bert/news

echo "POETRY"
mkdir /homes/etayl/code/bert/poetry
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/poetry/*" /homes/etayl/code/bert/poetry
python fix_data.py poetry 
vast5 sync "/homes/etayl/code/bert/poetry/*" s3://mobileye-team-angie/users/etay/nltk_data/poetry/
rm -rf /homes/etayl/code/bert/poetry

echo "quotes"
mkdir /homes/etayl/code/bert/quotes
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/quotes/*" /homes/etayl/code/bert/quotes
python fix_data.py quotes 
vast5 sync "/homes/etayl/code/bert/quotes/*" s3://mobileye-team-angie/users/etay/nltk_data/quotes/
rm -rf /homes/etayl/code/bert/quotes

echo "recipes"
mkdir /homes/etayl/code/bert/recipes
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/recipes/*" /homes/etayl/code/bert/recipes
python fix_data.py recipes 
vast5 sync "/homes/etayl/code/bert/recipes/*" s3://mobileye-team-angie/users/etay/nltk_data/recipes/
rm -rf /homes/etayl/code/bert/recipes

echo "reddit"
mkdir /homes/etayl/code/bert/reddit
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/reddit/*" /homes/etayl/code/bert/reddit
python fix_data.py reddit 
vast5 sync "/homes/etayl/code/bert/reddit/*" s3://mobileye-team-angie/users/etay/nltk_data/reddit/
rm -rf /homes/etayl/code/bert/reddit

echo "NEWS"
mkdir /homes/etayl/code/bert/news
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/news/*" /homes/etayl/code/bert/news
python fix_data.py news 
vast5 sync "/homes/etayl/code/bert/news/*" s3://mobileye-team-angie/users/etay/nltk_data/news/
rm -rf /homes/etayl/code/bert/news

echo "reviews"
mkdir /homes/etayl/code/bert/reviews
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/reviews/*" /homes/etayl/code/bert/reviews
python fix_data.py reviews 
vast5 sync "/homes/etayl/code/bert/reviews/*" s3://mobileye-team-angie/users/etay/nltk_data/reviews/
rm -rf /homes/etayl/code/bert/reviews

echo "tweets"
mkdir /homes/etayl/code/bert/tweets
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/tweets/*" /homes/etayl/code/bert/tweets
python fix_data.py tweets 
vast5 sync "/homes/etayl/code/bert/tweets/*" s3://mobileye-team-angie/users/etay/nltk_data/tweets/
rm -rf /homes/etayl/code/bert/tweets