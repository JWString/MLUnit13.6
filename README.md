# ML Unit 13.6: Survey Existing Research and Reproduce Available Solutions

Objective: 

For this unit I decided to focus on the popular gradient descent optimization algorithm, Adam, for the purpose of testing the extensibility of my neural net framework, and also I thought it would make for an interesting performance comparison against my own custom optimization algorithm. More detailed analysis & comparison will not be available until the functionality of my Capstone project is sufficiently completed, but some initial measurements have been included.

Published documentation for Adam:

https://arxiv.org/abs/1412.6980

Current performance over an XOR dataset with Adam:

![Adam test result](/docs/AdamResult.jpg)

Performance over an XOR dataset with custom optimization:

![Custom test result](/docs/CustomResult.jpg)

Update:

Additional testing revealed that I made a mistake in the decay factor modifications.  After correction, the performance with the updated Adam implementation is included below.

![Adam test result](/docs/AdamResult2.jpg)