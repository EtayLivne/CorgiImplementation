echo academic
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/academic/*" /homes/etayl/code/bert/local_dataset/baseline/academic
python fix_file_size.py academic academic 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/academic/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/academic/

echo code
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/code/*" /homes/etayl/code/bert/local_dataset/baseline/code
python fix_file_size.py code code 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/code/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/code/

echo judicial
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/judicial/*" /homes/etayl/code/bert/local_dataset/baseline/judicial
python fix_file_size.py judicial judicial 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/judicial/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/judicial/

echo medical
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/medical/*" /homes/etayl/code/bert/local_dataset/baseline/medical
python fix_file_size.py medical medical 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/medical/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/medical/

echo news
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/news/*" /homes/etayl/code/bert/local_dataset/baseline/news
python fix_file_size.py news news 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/news/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/news/

echo poetry
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/poetry/*" /homes/etayl/code/bert/local_dataset/baseline/poetry
python fix_file_size.py poetry poetry 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/poetry/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/poetry/

echo quotes
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/quotes/*" /homes/etayl/code/bert/local_dataset/baseline/quotes
python fix_file_size.py quotes quotes 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/quotes/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/quotes/

echo recipes
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/recipes/*" /homes/etayl/code/bert/local_dataset/baseline/recipes
python fix_file_size.py recipes recipes 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/recipes/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/recipes/

echo reddit
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/reddit/*" /homes/etayl/code/bert/local_dataset/baseline/reddit
python fix_file_size.py reddit reddit 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/reddit/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/reddit/

echo reviews
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/reviews/*" /homes/etayl/code/bert/local_dataset/baseline/reviews
python fix_file_size.py reviews reviews 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/reviews/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/reviews/

echo tweets
vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/tweets/*" /homes/etayl/code/bert/local_dataset/baseline/tweets
python fix_file_size.py tweets tweets 500
vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/tweets/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/b_500/tweets/


 rm -rf /homes/etayl/code/bert/local_dataset/upload_cache/