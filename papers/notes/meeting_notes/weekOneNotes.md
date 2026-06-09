Introduction to Machine Learning
- a computer learning patterns from data rather than being directly tasked for every single thing

Data = information the computer learns from 
Features = important pieces of the data 
Model = the brain that learns the pattern from the data 
training = teaching the model to understand the data 
prediction = the AI making a guess of what to do after learning

Internet Scale 
- more things to look and search more vast
GPUs everywhere
- 100 trillion operations per second
- no longer takes very long to train a model
Deep Learning Transfomers

AI: building systems that percieve reason learn and act in ways that as if a human did them
narrow AI
- does one thing well: ChatGPT, Tesla Autopilot
- Image classifer
- Spam filter
- Voice Assistant

general AI 
- system that matches a human across any cognitive takes
- reason across domains
- cpmmon sense
- self-improve
super AI 
- superpasses the best human smarter than humans in almost everything
rlly powerful AI tools that can solve: 
- code 
- equations 
- learn fast
- create things 
- think through a task almost like a human

how does the machine learn?
make a guess
check if it is wrong
adjust
try again and again and again and again

learning is really adjusting mathematical valuse to reduce error

deep learning models do this with ALOT of math called neural networks
- input goes in 
- layers process information
- output

transformers: learn patterns in language

supervised
with a teacher
input+correct anser

unsupervised
no answers given 
- ai groups the data to similar ones with what is given 
    - clustering

Training Set / Validation Set / Test Set
train: learning the material the model studies this data ana d adjusts itself 
v: are we improving or memorizinig?
test: model should have NEVER seen this data before( checks if AI truly learng the patterns) 

underfit: soo simple ( a slopefollowing th efirst few lines of a dot plot disregarding the rest)
gerneralized: just right (basically gets evey bit of the data)
overfit: too complex ( tracing every plot of dot plot)

the math behing it: 
- linear regression ( y= mx+b)
    - ai changes the slope and intercepts

using a loss function to measure error: 

mean squared error: 1/n sum of (y_true - y_pred)^2
mean absolute error: 1/n sum of |y_true - y_pred|

accuracy: coprrect pred/total predictions

optimization (gradient descent)
feel which direction goes downward
take a small step down
repeat over and over

That’s gradient descent.
optimization is the best weights to minimize error: lowest possible loss/error.

Recurrent neural networks 
- made for sequences 
good for: sentences, speeches, time-series data 

Long Short Term Memorys LSTMs: designed to remember information for longer 

generative adversarial networks: 2 neural networks competing against each other 
- generator creats fake data; discriminator checks if it is real or fake 

Large Language Models (LLMs)
massive transformer-based autoregressive models trained on HUGE amounts of text.
- chatgpt 
- writing ai 

autoregressive and diffusion models: 
predict on thing at a time based on previous outputs this is how they try and guess your sentences 
modern-image generation is diffusion models generates image from random noise 

Adversarial Attacks in Machine Learning & Autonomous Vehicles

ML models can be tricked even if it seems that everything else is working well 
Adversarial attacks are like purposely tricking the AI model by modifying the input data 
    small changes in the input data can confuse the model alot
    ex: 
        adding stickers to stop signs
        changing pixels in an image
        shining lasers at cameras
        using reflective patches
Humans still understand it but the AI model will not 

attacks include like the stop sign attack, facial recognition attacks where glasses or makeups, voice recognition attacks humans cant hear the sound but the AI model like siri can!

model-level v system-level success is like ok the camera is chopped but luckily the whole system didnt fail because i aint forget 
like if the attack works on 70% the framebut the car sees a stopsign for 30%

theyre saying before they focused too much on neural networks without also looking at how real autonomous commercial cars work 
so they rented a few cars and tested different commercial cars like mercedes tesla hyundai toyota honda implemneted the atacks drove toward the sing and saw what happenend

They tested:
    stop signs
    speed limit signs
    different attack types
    different neural network architectures

they saw that some attacks worked for some but not for others like there was no grand generalization or at least not a pretty one 

black box attack: 
    attacker doesnt know: 
    - model architecture
    - weight 
    - traning data 
only can give it its input and see the resulting output 
ex:
    Commercial Tesla system.
    Researchers don’t know the internal code,
    but they test how it responds.

white-box attack: 
attacker knows EVERYTHING
    - model architecture
    - weight 
    - traning data 
much easier to attack it bc u know what ur workin with
mostly used in reseach


targeted attack or untargeted attack
targeted: the attacker wants a specific wrong answer
untargeted: the attacker wants any wrong answer it just wants it to fail 

attacks show the weakness in the neural network and its sensitivity

NEURAL NETWORKS LEARN MATHEMATICAL PATTERNS
how the process works: 
researchers gather hella images like stop signs yield signs etc and that is the dataset 
clean data = normal trustworthy data like real things that are good
poisoned data is just malicious or corrupted data like adding fake things 

teh neural network learns the patterns from the data it adjusts weights using optimization gradient descent and loss functions 

security and privacy attack difference
security is used to try to make the AI fail like fool a self-driving car 
privacy attacks are used to steal information from the ai like ssn or credit card info

an evasion attack is just attacking after the model is already trained you are only changing the input not the entire model 

poisoning attacks are basically just attacvking during trainig poising the clean data by also adding teh poisoned data 

back door attack is a type of poisoning this is basically just attackers insert different trigger patterns during training 

Normal ML Pipeline
Clean Data →
Training →
Model →
Predictions

Poisoning Pipeline
Poisoned Data →
Training →
Corrupted Model →
Dangerous Predictions

Evasion Pipeline
Normal Model →
Malicious Input →
Wrong Prediction

membership inference attacks 
determinf whether something has been used during training 

math formulas to know for this: 
    linear regression (y=mx+b)
    mean squared error 
    MSE=n1​∑(ytrue​−ypred​)2

    gradient descent update rule: 
    wnew​=wold ​− α(dw/dL​)
        w = weights/parameters
        L = loss/error
        derivative tells direction to reduce error
        α = learning rate (step size)