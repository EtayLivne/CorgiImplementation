def gen_script(categories, target_categories, exp_name, new_file_size):
    
    strings = [
        f'echo {category}\n' +\
        f'vast5 sync "s3://mobileye-team-angie/users/etay/nltk_data/{category}/*" /homes/etayl/code/bert/local_dataset/baseline/{category}\n' +\
        f'python fix_file_size.py {category} {target_category} {new_file_size}\n' + \
        f'vast5 sync "/homes/etayl/code/bert/local_dataset/upload_cache/{category}/*" s3://mobileye-team-angie/users/etay/corgi_gpt_exps/{exp_name}/{category}/\n\n'
        for category, target_category in zip(categories, target_categories)
    ]
    
    strings = "".join(strings) + "\n rm -rf /homes/etayl/code/bert/local_dataset/upload_cache/"
    
    with open(f"/homes/etayl/code/bert/one_time_scripts/fix_file_sizes_{exp_name}.sh", "w") as handler:
        handler.write(strings)
    
    
d = {
    "academic": 14325,
    "code": 40000,
    "judicial": 4086,
    "medical": 4086,
    "news": 5068,
    "poetry": 3086,
    "quotes": 1179,
    "recipes": 34228,
    "reddit": 257,
    "reviews": 19816,
    "tweets": 8957
}

categories = list(d.keys())
new_categories = list(d.keys())

gen_script(categories, new_categories, "b_500", 500)