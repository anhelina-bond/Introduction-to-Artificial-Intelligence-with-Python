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
    all_pages = corpus.keys()
    connected_links = corpus[page]
    if len(connected_links) == 0:
        connected_links = corpus.keys()

    link_prob =  damping_factor / len(connected_links) 
    all_pages_prob = (1-damping_factor)/len(all_pages)

    distribution = dict()

    for p in all_pages:
        if p == page: 
            distribution[p] = all_pages_prob
        elif p in connected_links:
            distribution[p] = link_prob + all_pages_prob
        else:
            distribution[p] = 0
    return distribution
    
    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = list(corpus.keys())
    page_rank = {page: 0 for page in all_pages}

    page = random.choice(all_pages)

    for _ in range(n):
        page_rank[page] += 1

        probability = transition_model(corpus, page, damping_factor)
        weights = [probability[p] for p in all_pages]

        page = random.choices(all_pages, weights=weights, k=1)[0]

    #normalize ranks
    for p, rank in page_rank.items() :
        page_rank[p] = rank/n

    return page_rank
    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = list(corpus.keys())
    num_pages = len(all_pages)
    rank = 1/num_pages
    page_rank = dict()
    threshold = 0.001

    current_with_parents = dict()
    
    #initialize equal ranks
    for page in all_pages:
        page_rank[page] = rank

    while True:
        new_page_rank = {}

        for page in all_pages:
            # Calculate the sum of the PageRank contributions from pages that link to the current page
            sum = 0
            for p in all_pages:
                if page in corpus[p]:
                    sum += page_rank[p] / len(corpus[p])
                # If p has no outgoing links, treat it as linking to every page (including itself)
                elif len(corpus[p]) == 0:
                    sum += page_rank[p] / num_pages
            # Apply the PageRank formula
            new_rank = (1-damping_factor)/num_pages + damping_factor * sum
            new_page_rank[page] = new_rank

        # Check for convergence: If all PageRank values change by less than the threshold, stop iterating
        if all(abs(new_page_rank[page] - page_rank[page]) < threshold for page in all_pages):
            break

        # Update the page_rank for the next iteration
        page_rank = new_page_rank

    return page_rank

    raise NotImplementedError


if __name__ == "__main__":
    main()
