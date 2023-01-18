import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    probability = {}

    # check if page has outgoing links
    dict = len(corpus.keys())
    pages = len(corpus[page])

    if len(corpus[page]) < 1:
        # there are no outgoing pages, chooses randomly from all possible pages
        for key in corpus.keys():
            probability[key] = 1 / dict

    else:
        # there are outgoing pages, calculate probability
        random = (1 - damping_factor) / dict
        even = damping_factor / pages

        for key in corpus.keys():
            if key not in corpus[page]:
                probability[key] = random
            else:
                probability[key] = even + random

    return probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # start sample count at 0
    sample_count = corpus.copy()
    for i in sample_count:
        sample_count[i] = 0
    sample = None

    # go through loop number of times _ thats what this means
    for _ in range(n):
        if sample:
            # previous sample is available, used transition model
            tm = transition_model(corpus, sample, damping_factor)
            tm_lst = list(tm.keys())
            tm_weights = [tm[i] for i in tm]
            sample = random.choices(tm_lst, tm_weights, k=1)[0]
        else:
            # no previous sample, choosing randomly
            sample = random.choice(list(corpus.keys()))

        # count each sample
        sample_count[sample] += 1

    # turn sample count to percentage
    for i in sample_count:
        sample_count[i] /= n

    return sample_count


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_num = len(corpus)
    old_dict = {}
    new_dict = {}

    # assigning each page a rank of 1/page number
    for page in corpus:
        old_dict[page] = 1 / page_num

    # repeatedly calculating new rank values basing on all of the current rank values
    while True:
        for page in corpus:
            tmp = 0
            for link_page in corpus:
                # check if page links to our page
                if page in corpus[link_page]:
                    tmp += (old_dict[link_page] / len(corpus[link_page]))
                # if page has no link
                if len(corpus[link_page]) == 0:
                    tmp += (old_dict[link_page]) / len(corpus)
            tmp *= damping_factor
            tmp += (1 - damping_factor) / page_num

            new_dict[page] = tmp

        diff = max([abs(new_dict[x] - old_dict[x]) for x in old_dict])
        if diff < 0.001:
            break
        else:
            old_dict = new_dict.copy()

    return old_dict


if __name__ == "__main__":
    main()
