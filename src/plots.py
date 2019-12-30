import matplotlib.pyplot as plt
import json


def plot_histogram_for_source_report(path):
    with open('source_stat_2014_1.json', 'rb') as f:
        data = json.load(f)

    key = list(data.keys())[0]
    val = data[key]

    x = list(val.keys())
    y = list(val.values())

    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.title('number of times a news was followed by other sources for {} during 2014 Jan'.format(key))
    plt.tight_layout()
    plt.savefig('{}'.format(key))


def main():
    plot_histogram_for_source_report('./')


if __name__ == '__main__':
    main()

