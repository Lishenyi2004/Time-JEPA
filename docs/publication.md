# Preprint and code publication

Time-JEPA is a manuscript submitted to ICLR 2027 and is being prepared for release as an arXiv preprint. It has not been accepted. The companion repository is [Lishenyi2004/Time-JEPA](https://github.com/Lishenyi2004/Time-JEPA). Authors, in order: Shenyi Li, Ziyang Zhang, Wen Wu, Ming Liu, Chao Zhang. Publication fields are recorded in [publication.json](../publication.json); the arXiv identifier is pending submission.

## Publish the companion code

Use the latest clean directory built by [release.md](release.md), rather than the original experimental repository or its Git history. The public package includes source code, selected scientific configurations, local preprocessing tools, a bounded quickstart and a resource index. Obtain datasets from their original providers. No processed dataset will be hosted on Hugging Face as part of this release plan.

The confirmed repository URL and author list are recorded in README and `CITATION.cff`; project and upstream notices are retained in `NOTICE` and the existing license files. After arXiv assigns an identifier, add the actual preprint citation and link. Regenerate the experiment index when its inputs change, refresh the checksums of intentionally changed published files, and rebuild the clean package. Dataset metadata has a separate scope described in [DATA-LICENSES.md](../DATA-LICENSES.md).

## Publish the preprint

Prepare the arXiv version with the real authors and affiliations, and link the public GitHub repository from the paper. After arXiv assigns an identifier, add the paper URL and a preprint citation to README and create `CITATION.cff` using the confirmed author list. Use the actual preprint publication year; the ICLR 2027 submission year does not make the work an accepted 2027 conference paper.

Suggested BibTeX structure, with placeholders to replace before use:

```bibtex
@misc{timejepa,
  title         = {Time-JEPA: Grounding Video World Models with Physical Time Series},
  author        = {Shenyi Li and Ziyang Zhang and Wen Wu and Ming Liu and Chao Zhang},
  year          = {PREPRINT PUBLICATION YEAR},
  eprint        = {ASSIGNED ARXIV IDENTIFIER},
  archivePrefix = {arXiv},
  primaryClass  = {ACTUAL ARXIV CATEGORY},
  url           = {ACTUAL ARXIV URL}
}
```

## Reviewing version

ICLR 2027 allows arXiv publication during review, while the submitted paper and supplementary material must remain anonymous. If providing code to reviewers, use a separately anonymized supplementary archive or anonymous repository; inspect links, acknowledgements, author fields and file contents in that reviewing copy. Do not link an author-identifying public repository directly from the anonymous submission.

Source: [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), checked on 2026-10-13. This local preparation does not upload a paper, create a remote repository or publish data.
