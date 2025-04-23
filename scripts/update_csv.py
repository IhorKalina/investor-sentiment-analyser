import csv

if __name__ == '__main__':
    with open("crude_oil_reddit_posts_updated.csv") as f:
        r = csv.DictReader(f)
        new = []
        for x in r:
            new_x = x.copy()
            new_x["Crude oil index closed"] = str(round(float(x["Crude oil index closed"]), 2))
            new.append(new_x)


    with open("crude_oil_reddit_posts_updated_1.csv", mode="x") as f:
        r = csv.DictWriter(f, fieldnames=list(new[0].keys()))
        r.writerows(new)