import json
import pandas

# merge duplicate and equal categories
# remove some corner topics

dataset = []
for line in open('data/News_Category_Dataset_v2.json', 'r'):
    dataset.append(json.loads(line))

newDataset = []
for index, item in enumerate(dataset):
    category = item['category']

    # remove spams
    # MONEY! TECH
    if category == "PARENTING" or \
            category == "PARENTS" or \
            category == "RELIGION" or \
            category == "ENVIRONMENT" or \
            category == "FIFTY" or \
            category == "GREEN" or \
            category == "MONEY":
        continue

    # normalize categories
    if category == "WELLNESS" or category == "HEALTHY LIVING":
        item['category'] = "HEALTHY"

    elif category == "STYLE":
        item['category'] = "STYLE & BEAUTY"

    elif category == "TASTE":
        item['category'] = "FOOD & DRINK"

    elif category == "ARTS" or category == "ARTS & CULTURE" or category == "CULTURE & ARTS":
        item['category'] = "ARTS & CULTURE"

    elif category == "COLLEGE" or category == "EDUCATION" or category == "SCIENCE":
        item['category'] = "EDUCATION & SCIENCE"

    elif category == "QUEER VOICES" or \
            category == "THE WORLDPOST" or \
            category == "WEIRD NEWS" or \
            category == "WORLDPOST" or \
            category == "BLACK VOICES" or \
            category == "WORLD NEWS" or \
            category == "GOOD NEWS" or \
            category == "LATINO VOICES":
        item['category'] = "NEWS"

    newDataset.append(item)

pandas.DataFrame(newDataset).to_csv("data/news-clean-category-1.csv")
