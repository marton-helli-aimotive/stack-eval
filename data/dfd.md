# Dataset Documentation

## Motivation

1. **For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.**
   - The dataset was created to evaluate the performance of Large Language Models (LLMs) in various coding assistance tasks, including code writing, debugging, code review, and answering conceptual questions. It aims to provide a comprehensive benchmarking suite named StackEval for this purpose.

2. **Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**
  - Nidhish Shah, Zulkuf Genc, Dogu Araci from the ProsusAI team.

3. **Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.**
  - N/A

4. **Any other comments?**
  - None

## Composition

5. **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.**
  - Dataset contains questions for evaluation of coding-assistant tasks across multiple programming languages.

6. **How many instances are there in total (of each type, if appropriate)?**
  - StackEval: 900 questions, StackUnseen: 300 questions

7. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).**
  - Dataset is a sample of the StackOverflow public dump, sampled based on the StackOverflow developer survey.

8. **What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.**
  - The dataset consists of labelled text data.

9. **Is there a label or target associated with each instance? If so, please provide a description.**
  - Yes, label indicates the accepted answer for each question.

10. **Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.**
   - N/A.

11. **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? If so, please describe how these relationships are made explicit.**
   - N/A

12. **Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.**
   - N/A.

13. **Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.**
   - The labels are human curated, so standard issues in interlabel agreement may arise, but have been mitigated by the use of a 3rd domain expert.

14. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.**
   - N/A.

15. **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.**
   - N/A.

16. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.**
   - N/A.

17. **Does the dataset relate to people? If not, you may skip the remaining questions in this section.**
   - N/A.

18. **Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.**
   - N/A.

19. **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.**
   - The questions are sampled from StackOverflow. Specifically seaching for said questions may reveal the individual's identity on StackOverflow.

20. **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.**
   - N/A.

21. **Any other comments?**
   - None.

## Collection Process

22. **How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.**
   - See the section on Dataset curation of the paper.

23. **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?**
   - See the section on Dataset curation of the paper.

24. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**
   - See the section on Dataset curation of the paper.

25. **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**
   - See the section on Dataset curation of the paper.

26. **Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.**
   - The StackEval dataset was sampled between January 2018 - September 2023. The StackUnseen dataset was sampled between September 2023 - March 2024.

27. **Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.**
   - N/A.

28. **Does the dataset relate to people? If not, you may skip the remaining questions in this section.**
   - N/A.

29. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?**
   - N/A.

30. **Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.**
   - Yes.

31. **Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.**
   - Yes.

32. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).**
   - Yes.

33. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.**
   - N/A.

34. **Any other comments?**
   - None.

## Preprocessing/Cleaning/Labeling

35. **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.**
   - Yes, see the section on Dataset curation of the paper.

36. **Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.**
   - N/A.

37. **Is the software used to preprocess/clean/label the instances available? If so, please provide a link or other access point.**
   - Yes, we provide the code for generating evaluations in [evaluation](/evaluation.py).

38. **Any other comments?**
   - None.

## Uses

39. **Has the dataset been used for any tasks already? If so, please provide a description.**
   - Yes, the dataset has been used to evaluate multiple LLMs, the details of which are available in the paper.

40. **Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.**
   - N/A.

41. **What (other) tasks could the dataset be used for?**
   - None.

42. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks)? If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?**
   - No.

43. **Are there tasks for which the dataset should not be used? If so, please provide a description.**
   - Since the proposed dataset is a benchmark, future LLMs should not be trained on these benchmarks to preserve their integrity.

44. **Any other comments?**
   - None.

## Distribution

45. **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.**
   - Yes, the dataset is freely available on the GitHub repo.

46. **How will the dataset be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?**
   - Yes, the dataset is freely available on the GitHub repo.

47. **When will the dataset be distributed?**
   - Yes, the dataset is freely available on the GitHub repo.

48. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.**
   - Yes, the dataset is released only for research purposes, and shall not be used for commercial use.

49. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.**
   - Yes, the dataset is released only for research purposes, and shall not be used for commercial use.

50. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.**
   - None.

51. **Any other comments?**
   - None.

## Maintenance

52. **Who will be supporting/hosting/maintaining the dataset?**
   - The original authors.

53. **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**
   - The authors can be contacted by filing and issue on GitHub.

54. **Is there an erratum? If so, please provide a link or other access point.**
   - N/A.

55. **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?**
   - The StackUnseen dataset will be updated every six months. In case of errors, please file an issue on GitHub.

56. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.**
   - N/A.

57. **Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users.**
   - Yes, available on GitHub.

58. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.**
   - Yes, others may build upon the dataset through explicit permission of the original authors, and giving credit in all derivative works. The derivative works may not be used for commericial purposes.

59. **Any other comments?**
   - None.