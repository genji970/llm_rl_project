# recent-llm-paper-review-coding-contrastive-preference-optimization-
code implementation in colab env for data prerprocessing and model fine tuning parts. colab 환경에서 데이터 전처리 ~ model 파인튜닝까지의 과정

## the implementation of the fine-tuning process for the model and data preprocessing ##

## title : "contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation" ##

authors : Haoran Xu♠ AmrSharaf♡ YunmoChen♠ WeitingTan♠ Lingfeng Shen♠ Benjamin Van Durme♠ Kenton Murray∗ ♠ Young Jin Kim∗ ♡



![손실함수](https://github.com/user-attachments/assets/bcebf79b-11c1-47e7-bc04-f422e7d84ef1)

constraint is the form of kl divergence. by such constraint, the gap between perfect model ‘s policy(given conditional, y_preferred | x) and model’s policy(y_preferred | x) goes to value less than epsilon.(epsilon : for any positive value bigger than 0) epsilon is not a certain value. so comparison between single val and another is not right but I just used expression.

Because kl_divergence sometimes outputs discontinuity, this paper suggests to set lagrangian multiplier in front of kl divergence term. -> soft way 

Let’s look at appendix c,

![스크린샷 2024-12-02 011116](https://github.com/user-attachments/assets/c820be17-dfbc-4c9c-887d-1e2775cdce2e)


Right formula means L_preferred + kl divergence term when lagrangian multiplier is 1.

![전체함수2](https://github.com/user-attachments/assets/dcdf25cd-7a76-496a-aa8c-d77f57701b55)

NLL : negative log likelihood


This paper suggests focusing on contrastive learning using output_probability from preferred data and output_probability from dis preferred data rather than using only gold reference. Since, gold references are not always gold.

My thought : since, context varies according to different situation, gold reference has limit. And this paper also mentioned memory inefficiency and speed inefficiency. Since, policy for preferred data and policy for dis preferred data are double counted(as far as I understand). I think it is reasonable. Since, as mentioned above, gold reference depends on situation. There’s no perfect intrinsic value in real world. 

contrastive learning in this paper seems ensemble. Meaning data's feature is crucial. On the other hand, model's logit contains distribution with vicab_size shape. It somehow works on contarstive loss.

## data preprocessing ##

## custometrainer build ##
cpo loss :
nll loss : -log(policy_preferred)
kl term :

## experiment result ##
xcomet : 
