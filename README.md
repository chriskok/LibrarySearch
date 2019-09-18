# LibrarySearch

## Running live book search

Install the necessary libraries with pip

Then run the command:

```
python find_books_BF.py
```

## Notes
- Drew a lot of inspiration from the 3-level content based image retrieval (CBIR) system proposed here: https://towardsdatascience.com/judging-a-book-by-its-cover-1365d001ef50

## Scraping images

Install:

https://github.com/hardikvasa/google-images-download

Then run:
```
googleimagesdownload -k "oreilly books" -f jpg -i "oreilly" -l 100
```
