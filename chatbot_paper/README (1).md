# Chatbot_Reduction-in-execution-time_with-reference-to-paper-Enhancing-Robustness-in-LLM-
Chatbot_Reduction in execution time_with reference to paper "Enhancing Robustness in Large Language Models : Prompting for Mitigating the Impact of Irrelevant Information"

###

If there's any error, plz tell me. I'm looking forward to it !!!!

###

This project gain ideas from recent paper(2024.08) from archive.

authors : Ming Jiang , Tingting Huang , Biao Guo, Yao Lu, Feng Zhang


Main difference : 
This paper used chatgpt3.5                          ,
I used chatgpt4 api

This paper suggests experiment with irrelevant data and tries to figure out the way for model to filter irrelevant words  ,   
I used data's context , question , answer , con as embedding vectors meaning context.

This paper found impact of irrelevant information                     ,
I tried to figure out if embedding space indicates irrelevant vectors , then will model answer fastly than just putting normal original question?


Experiment result : I used just one question. And found out 

1) original query + no context takes 5.01 seconds
2) original query + context takes 4.79seconds
3) (original query + irrelevant data) + no context takes 8.86 seconds
4) (original query + irrelevant data) + context takes 6.23 seconds
