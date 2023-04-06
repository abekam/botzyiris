n this

video we're going to be covering all of

the lane chain Basics with the goal of

getting you building and having fun as

quick as possible my name is Greg and

I've been having a ton of fun building

out apps in langchain now I share most

of my work on Twitter so if you want to

go check it out links in the description

you can go follow along with me now this

video is going to be based off of the

new conceptual docs from lanechain and

the reason why I'm doing a video here is

because it takes all the technical

pieces and abstracts them up into more

theoretical qualitative aspects of Lane

chain which I think is extremely helpful

for it and in order to understand this a

little bit better I've created a

companion for this video and that is the

Lang chain cookbook links in the

description if you want to go check that

out please go and check out the GitHub

and you can follow along here I'm gonna

put a lot of time stamps in the

description as well there's gonna be a

fair amount of content in this one so

you can watch it all the way through or

if you want to skip to a certain section

feel free to jump to that time stamp all

right without further Ado let's jump

into it all right here are the new

conceptual docs from Lang chain now the

reason why these are different is

because there are the python docs which

are going to be the more technical

focused one or the JavaScript docs as

well which is also more technical

documentation however these concepts are

more qualitative so you can understand

what is going on in the background of

these different sections here now we're

going to focus on these components of

Lang chain there's an entire section on

use cases which is when you actually put

these into practice and that is going to

be a part two of this video so we won't

jump into this today that would be a too

long for us we're going to run through

schema models prompts indexes indexes

memory chains and agents with a working

code sample for each one of those well

without further Ado let's jump into some

code here we are with the Lang chain

cookbook now my goal is to make this a

dense document with a ton of links so

you can go and self-service right into

their links in the description and if

you want to follow along I encourage you

to get this on your computer and go from

it go for it from there so the goal of

this dock is to provide an introductory

understanding of the components and use

cases of Lang chain in a explain like M5

way with examples and code Snippets for

use cases check out part two which is

not made yet that is coming soon

hopefully by the time you see this it

will be a bunch of links here we go into

what is Lang chain so Lang chain is

going to be a framework for developing

applications powered by language models

well Greg openai just came out with

plugins yes but there is a whole lot of

other things that you can do with

language models outside of those and

Lang chain helps abstract a ton of that

so that you're able to work with it more

easily and intermix different pieces and

customize really how you need to

so Lane chain makes the complicated

parts of working and building with AI

models easier it does this in two main

ways the first big way is going to be

through integration so you can bring

external data such as your files other

applications API data to your language

models which is cool the other big way

that it helps do this is through agency

so it allows your language models to

interact with its environment via

decision making basically you're using

the language model to help decide which

action to take next and you do this when

the path isn't so clear or it may be

unknown and we'll get into more more of

that later

so why link chain specifically there are

four big reasons why I like Lane chain

the first one is going to be for the

components Lang chain makes it easy to

swap out abstractions and components

necessary to work with language models

basically they've created a ton of tools

that make it super simple to work with

language models like chat GPT or

anything on how you face how you may

want also because it allows you to

customize chains really easily so

there's a ton of out of the box of

support for using and customizing chains

basically combining series of actions

together

on the qualitative side of why LinkedIn

is awesome is because the speed is great

almost every day I need to go and make

sure that I'm on the latest branch of

Lang Chan and I go and I update it every

time so the speed is awesome the other

really cool part is the community so

there's a ton of meetups there's a

Discord Channel and there's a ton of

events like webinars that go on

throughout the week that are really

awesome learning resources for us

cool now again to summarize all this why

do we need Lang chain well because

language models can be pretty

straightforward it is text in text app

and you may have experienced this

yourself however once you just start

developing applications there's a ton of

friction points that Lang chain is going

to help you develop uh there's a ton of

friction points that they're going to

help you with basically now the last

thing that I'll say about this before we

jump into it is that this cookbook isn't

going to cover all of the aspects of

Lang chain this isn't meant to be a

replacement for the documentation online

this is meant to show you a very broad

overview about the capabilities that

there are with my interpretation of them

and my voice over with it and with that

I'm hoping that you can get to building

and impact as quick as possible

I'm super curious to see what you build

so please let me know and uh I will

hopefully uh I would love to see it

first thing we're going to do is we're

going to import our openai API key now I

have a hidden cell here but you're going

to replace your API key API key right

here just throw that in there the first

aspect of link chain components that

we're going to look at is the schema now

I almost didn't even include this one

but the first one is going to be text

now what's really cool about these

language models is that text is the new

programming language not verbatim not

per se but we're using a lot more

English language to tell language models

what to do in this case what day comes

after Friday is an example of something

I may go tell a language model and it is

going to respond back to me with a

natural language response very cool next

up is going to be chat messages so like

text chat messages are similar but they

have different types the first type is

going to be system and this is helpful

background context that tell the AI what

to do all right like your helpful

teacher assistant bot or something then

we have human messages and these are

messages that are intended to represent

the user and so literally user input or

something that I may text from it then

we have ai messages and these are

messages that show what the AI responded

with and the cool part about this is the

AI may or may not have actually

responded with it but you can tell it

that it did so that it has additional

context on how to answer you okay so

what I'm going to do here is I'm going

to import chat open Ai and my three

message types and then I'm going to

create my chat model I'm going to do

that and then I'm going to type in two

messages the first system message is you

are a nice AI bot that helps a user

figure out what to eat in a short

sentence and then a human message I like

tomatoes what should I eat let me go

ahead and run this

and you get an AI message back because

this is what it responds with you could

try making a tomato salad with fresh

basil and mozzarella cheese thanks AI

That's cool what you can also do is you

can also pass more chat history and get

responses from the AI so in this case

you're a nice AI bot that helps a user

figure out where to travel to in one

short sentence

I'm saying I like the beaches where

should I go

I'm telling it that it responded to me

it didn't actually do this but I'm

telling it that it did you should go to

Nice France cool what else should I do

when

when I'm there and so the reason why I

did this one is because you'll notice

that I didn't say where I went it's

going to have to infer from the history

on what where I went and it says wow and

nice and so it picked up where I was

because it gets the history of the chat

messages now if you're making a chatbot

you could see how you could append

different messages that have been back

and forth uh I'm not sure if that's a

verb but back and forth through the user

okay

the next model that we're the next model

that we're going to look at is going to

be documents so documents are important

because this represents a piece of text

along with Associated metadata now

metadata is just a fancy word for things

about that document and in this case

this document or the text is held within

a field called page content so this is

my document it's full text that I've

gathered from other places awesome and

then I'm going to pass in some metadata

and this metadata is a dictionary of key

value pairs my document ID which is my

key here and then some random document

ID here that happens to be an INT it

could be whatever you want it to be my

document Source this is the Lang chain

papers and then my document create time

is going to be some timestamp whatever

you want it to be and this is going to

be can be whatever format you want this

is extremely helpful for when you're

making a large repositories of

information and you want to be able to

filter by it so instead of just going

and asking link chain to look at all

your documents in your database you can

go ahead and filter these by a certain

metadata go ahead and run this and you

can see here I get a document object

with a bunch of metadata on it from

there cool if those are the schemas that

we work with the next thing we're going

to look at is the different models now

these are the ways of interacting with

well different models

um but the reason why this is important

is because they're different model types

let me just show an example here the

normal one that we're looking at is

going to be the language model and this

is when text goes in and text comes out

okay now the first thing I'll do is I'll

import open Ai and I'll make my model

and you'll notice here that I changed my

model in case you ever want to change

your model as well and so I'm going to

pass in a regular string into this one

into my language model what day comes

after Friday

go ahead and run this and I get Saturday

comes out the other end but not all

models are like this you actually have

chat models as well and we looked at

this in the previous example but I

didn't call it out specifically so for

this one I'm going to import chat open

AI I'm going to import my messages again

I'm going to put temperature equals one

which means the model is going to get a

little spicy on me no but really it just

means it's going to have more creativity

and it's it's a little bit more

exaggerated and so in this case I'm

going to say you are an unhelpful AI bot

that makes jokes at whatever the user

says and in this case the user says I

would like to go to New York how should

I do this I'm going to go ahead and run

this model

you could try walking but I don't

recommend it unless you have a lot of

time on your hands

maybe try flapping your arms really hard

to see if it can fly there so as you can

see it took that system message

and it understood those directions and

it uh it wasn't very helpful for me well

because I told her not to be very

helpful

the last type of model that we're going

to look at is going to be your text and

betting model the reason why this one is

important is because we do a lot of

similarity searches and a lot of

comparing texts when working with

language models now in this case openai

also has an AI embeddings model that

we're going to use there's a lot of

embedding models out there you can use

whatever you want I just use open AI

because it feels like it's a standard

and it's very simple right now so I'm

going to pass in my API key I'm going to

get my embeddings engine ready and then

I'm going to define a piece of text hi

it's time for the beach let me go ahead

and do that text

and what I'm going to do is I'm going to

pass that text and I'm going to embed

that text so what that means is is it's

going to take this string which is just

a series of letters and it's going to

convert it into a vector and in this

case a vector is just simply a

one-dimensional array meaning a list of

numbers and that'll be a semantic

representation of that text that's a

fancy way of saying is that meaning of

that text is going to be embedded in

those numbers right there which makes it

really easy to compare

across other as others as well so I'm

going to put that in a variable called

text embeddings I'm going to see how

long my text embeddings is and I'm going

to get a preview of it so you'll notice

here that my text embedding length is

1536 this means that there are 1536

different numbers within that list that

represent the meaning of my text

that's a lot of numbers and I'm glad I

don't have to deal with them I'm glad

the computer can so here's a sample of

what those look like in case you're

curious I only show the first uh five

here but I put a dot dot dots you know

that there are

1531 other numbers out there next let's

look at prompts so prompts are going to

be the text that you send over to your

language model we've already sent some

prompts over to the language model but

they've been pretty simple in this case

we're going to start doing more

instructional prompts and passing those

to our model so again a prompt is what

we pass to our language model I'm going

to import open AI in this case I'm using

DaVinci as my model and I'm going to say

prompt equals this string now I use

three three double quotes because

um well I think it looks fancier no but

really it's just easier to use which is

why I like it in this case I'm not doing

anything fancy and I could have passed

this string right within my language

model but in this case I

made a variable for it because it's a

little bit easier to understand so

today's Monday tomorrow's Wednesday what

is wrong with the statement the

statement is incorrect tomorrow's

Tuesday not Wednesday so you can see how

it picked it up from there now why

prompts are cool is because we start to

get into the prompt template world

the reason why prompt templates are

important is because most of the time

you're going to be dynamically

generating your prompts meaning they

won't just be static strings that you

type out but you're actually going to be

inputting tokens or inputting

placeholders based off of the scenario

that we're you're working with

so in this case what I'm doing here is

I'm importing my packages again in this

case prompt template is going to be the

new one I'm going to do DaVinci again

okay great and in this case I'm going to

create a template to start so I really

want to travel to location you'll notice

my opened and closed brackets around

location which means that this is going

to be a token that I'm going to be

replacing later what should I do there

respond in one short sentence because

we're also just responded too much I'm

going to create a prompt template in

this case I'm going to put it in this

variable prompt my input variable is

going to be location which matches the

same name that we had up here and then

the template is this this whole thing

that I had here the final prompt is

going to be prompt.format which means

going to insert the values I tell you go

and insert the value Rome into where it

says location right here let's go ahead

and run this

so final prompt I really want to travel

to Rome which replace location up above

and here we have our prompt template

that's finally filled out and then in

terms of the output it tells me what I

should do so it took that information in

with Rome and responded one short

sentence it gives me this which is cool

all right the next cool part that we're

going to look at is the example

selectors so often when you're

constructing your prompts you're going

to do something called in context

learning this means that you're going to

show you're going to show the language

model what you want it to do and one of

the main ways that people do this is

through examples this could be about how

to answer a customer service request or

it could be how to respond to some

nuanced question and in this case I'm

going to pick examples however we have

example selectors because say you had 10

000 different examples you don't want to

throw all those into your uh into your

prompt they may not fit and they may not

be as relevant so you want to select

which ones you want and in this case

what I'm going to do is I'm going to

import a lot of things here but the one

I'm the main star of the show is going

to be the semantic similarity example

selector that's a long name for a

functionality that's going to select

similar examples so I'm going to get my

language model going again I'm going to

get my example prompt and this is just a

prompt template like we saw up above and

then I'm going to define a list of

different examples so in this case I

want to name a noun and then I want the

language model to tell me where this

noun is usually found so in this case a

pirate on a ship a pilot on a plane

driver in a car a tree

oh that's not true a tree in the ground

or a bird in a nest so I'll go ahead and

run that one

and then what we're going to do is we're

going to get our example selector ready

so we have our similar example selector

we're going to pass it the list of

examples that I just defined above but

then we're also going to pass it our

embedding engine and the reason why we

do this is because we're actually going

to match examples on their semantic

meaning so not just matching them off of

similar strings but off of what they

actually mean so in this case we're

going to use the open AI embeddings

which is one of the models that has been

shared by Facebook which is really cool

and this is going to help store our

embeddings and then we're going to tell

it

um how many we want how many examples we

want back in this case I want k equals

two let me go ahead and run that and

then we're going to have a new prompt

template here and this is going to be

the few shot prompt template meaning the

few shot part means that there's going

to be a few examples in there for us so

we give it our example selector we give

our example prompt which we made up

above and then we're going to add on

just some little strings before and

after to make it easier for the model so

give the location that an item is

usually found in cool and then the

suffix will be the input and the output

that we have from here based off of what

the user inputs then the input variable

go ahead and do that so here I'm going

to say my noun is student So based off

of the noun of student it's going to go

and find me the examples up above that

are most closely related to student and

we're going to use those examples so if

I would go ahead and do this I'm going

to say print and it's going to print me

the prompt that we're actually going to

use within or give to our language model

in this case it found the driver and it

found the pilot one being most similar

to student which is cool now if I were

to do a different one say flower it's

going to give me the the tree and the

bird examples okay but I'm going to

stick this with student and what I'm

going to do is I'm going to take this

prompter that we just made and I'm going

to pass that into the language model and

all of a sudden you get classroom the

next thing we're going to look at is

output parsers now that's kind of a

complicated way to say

um we need some structured output like

we want the language model to return a

Json object back to us why well because

it makes it a heck of a lot easier to go

deal with and work with on the other

side

there's two big Concepts when we talk

about output parsers first it's going to

be the formatting instructions piece so

this is the prompt template that is

going to tell your language model how to

respond back to you and Lang chain

provides us some conventions to do this

automatically which is cool and then the

second thing we're going to have is

going to be the parser and so this is

going to be the tool that is going to

parse the output of your language model

so the language model can only return

back a string but if we want a Json

object well we need to go and parse that

string and extract the Json Json from

that okay so we're going to get a

structured output parsing and we're

going to get the response schema from

there let's import our language model

again and we're going to have a response

schema so in this case I just want it to

be a two field Json object it I'm going

to have a bad string which is a poorly

formatted user input string and then a

good string this is your response a

formatted response and so the really

nice response from the um from the

language model there and in this case

I'm going to go ahead and create my

output parser which is going to read the

response schema and it's going to be

able to parse it for us but we won't use

that until just a second here

so first thing we're going to have is

our format instructions so on the output

and parser we're going to say get format

instructions and then let's print those

out

in fact I don't need to do that I could

just print this out directly right here

cool and so this is a piece of text that

is going to be input or insert put into

the prompt the output should be a

markdown code snippet format it in the

following schema Json and then the two

fields that I input up above but it did

the formatting for me or at least Lane

chain did for him to put it into here

so let's go ahead and create a prompt

template we're going to do a placeholder

variable for our format instructions and

then we're also going to do a

placeholder for user input this will be

the poorly formatted string that the

user is going to input and then finally

I put your response here just to tell it

it's like hey I'm done telling you

instructions give me a response we go

ahead and we get the prompt template we

have the user input we have a partial

variable of format instructions and this

will be the format instructions we had

up above we have our template which is

the string up above here and then we

have our prompt value so this will be

the actual value that is filled out with

the variables I tell it and I'm going to

say welcome to California with an

exclamation point let's go ahead and do

that one and here I print out the final

prompt that is going to be sent to the

llm we have user input Welcome to

California with everything we had up

above let's go ahead and run this

let's see what it responds back to us so

we get a string here it kind of looks

like gobbledygook but if we were to

print this out it'd make more sense but

before printing out let's just go ahead

and parse this and now we can actually

parse this and we get a nice uh Json

object back well in this case it's going

to be addict but um you can see here

it's typed

the next thing we're going to look at is

different indexes so in this case we're

going to be structuring documents in a

way that language models have a better

time working with them and one of the

main ways that lanechain does this it's

going to be through document loaders now

this is very similar to the open AI

plugins that just were released however

there's a lot of support for a lot of

really cool data sources in langchain

that aren't yet supported within the

plug-in World in this case I'm going to

be doing a Hacker News data loader so

all I'm doing is just passing a simple

URL to this data loader I'm going to say

hey go get me that data and so I'm

asking hey how many pieces of data did

you find

uh and in this case it found 76

different comments within this Hacker

News Post and I asked it to print me out

a sample and here we see uh one of the

responses by the moderator dang within

uh Hacker News and we see the response

there we see different comments uh you

can go and work with these within your

language model now which is pretty cool

another big piece of what we do a ton of

is text splitting so oftentimes your

document like your book or your essay or

whatever is going to be too long for

your language model you need to split it

up into chunks and text Splitters will

help with this now the reason why you do

this is because if you want a single

answer out of a book it wouldn't behoove

you too much to input that entire book

Into The Prompt one because it's too

long but two is because the signal to

noise ratio is too much or it's too

little for your language model to

effectively do its job it'd be a lot

better if you just put in a few pieces

of text into there and in order to get

those few pieces of text we need to do

splitting or chunking of those so in

this case I'm going to do text splitting

and the one that I use most often is

going to be the recursive character text

splitter there's a lot of different

types of text Splitters depending on

your use case I encourage you to go

check those out and in this case I'm

going to pull in a Paul Graham essay his

worked essay this one is quite long it

may be as long as actually so if I were

to read his document

I just have one big long document right

now which means it's a really long piece

of text but in this case what I want to

do is I want to have the recursive

character text splitter and I'm going to

say chunk size equals 150. this means

that I'm going to have a size of 150

when I end up splitting my star document

there and if you want chunk overlap that

means that the Venn diagram of your docs

is going to overlap just a little bit I

encourage you to play with these

variables to see which one works best

for your use case normally I wouldn't do

150 I'd probably do a thousand or two

thousand but for demonstration purposes

I'm doing 150. go ahead and run that and

so we had one document up above but

after I split it I now have 606

documents all right and if I wanted to

preview those I can go ahead and preview

these and see how they're nice and small

they're super small and if I wanted to

make this 50 for example well then my

chunks will be a whole lot smaller but

let me go ahead and make that bigger the

next thing we're going to look at is

going to be retrievers now retrievers

are easy ways to combine your documents

with your language models there's going

to be a lot of different types of

retrievers and the most widely supported

one is going to be the vector store

Retriever and it's most widely supported

because we're doing so much similarity

search within embeddings let's look at

an example here we're going to load up a

hologram essay just like how we had

beforehand I'm going to do some

splitting of it and so we're going to

get a whole bunch of documents

we're going to split the documents and

then I'm going to create embeddings out

of those documents and so all those

little chunks we're going to create

vectors out of them which is the

semantic meaning of them and then I'm

going to store those vectors within a

document store here okay and I'm going

to call that within a my DB there and

then I'm going to say hey this retriever

is going to be the DB but we're going to

set it as the retriever okay so it knows

to go get stuff and if I were to look at

this you can see here that we have our

Vector store retriever that's output

right here

okay we're going to take our Retriever

and I'm going to say hey go get me the

relevant documents what types of things

did the author want to build now in the

background what it's doing here is it's

taking the string and it's converting it

to a vector it's taking that vector and

it's going to go compare it to the

vector store that you have and find the

similar documents that come from there

so what I'm going to do here is I'm just

going to print out this is a one-liner

kind of complicated one just to print

out the preview of the documents that we

have here I'm just going to have it

print out the first two

docs is not defined great let's go ahead

and run those so all of a sudden these

are the previews of the docs that it

found there

um what I wanted was to not just build

things but build things that would last

so you can see here that out of all

those documents that I found it found

the two that were most similar to what I

was looking for which is really cool I

wanted to build things nice next let's

look at Vector stores so we briefly just

talked about Vector stores right before

this but to go into it a little bit

further think of a vector store really

the way that I think about it is a table

with rows with your embeddings and

Associated metadata that comes with it

an example of it is right here two main

players in the space are now are going

to be pine cone and weeviate however if

you want to you can go check out open

ai's retriever documentation and they

list a whole bunch of other ones that

you may find awesome for you

okay so let's go ahead and look at these

again we're going to import our models

we got our embeddings okay cool now with

these embeddings I'm gonna look at that

and based off of how I split my document

up above with a thousand chunks or a

thousand as a chunk size we get 78

documents Auto programs worked essay

okay what I'm going to do is I'm going

to create those as embeddings and I'm

going to get my embeddings list from

there and I'm going to let's look at the

length of the embedding list I have 78

embeddings reason why is because I have

one vector for each one of my documents

so all right makes sense and here's a

sample of one so here's an example of

what the embedding would look like it's

a numerical representation of the

semantic meaning of your document there

so your vector store is going to be

storing your embeddings and it makes

them easily searchable so in this case

it is going to take my embedding and

it's going to store it like a database

the next topic I want to look at is

going to be Memory so this is going to

be how you help your language models

remember things the most common use case

for this is going to be your chat

history so if you're making a chat bot

then you can tell up the history

messages that you've had beforehand

which makes it a whole lot better at

helping your user do whatever it needs

to do so in this case I'm going to

import chat message history and I'm

going to import my chat open AI again

and so I'm going to create my chat model

and then I'm going to create my history

model and to my history model I'm going

to add an AI message Hi and then I'm

going to add a user message what is the

capital of France so let me go ahead and

run that and if I were to take a look at

my history messages I get my two that

are input right there they're in the

right order as we would expect there to

be

so what's cool is that I can pass my

history of messages to the language

model and so in this case it is going to

read oh I said hi to start and then the

human message was what's the capital of

France and let's see what it responds

back to us the capital France is Paris

and it gives us an AI message which is

cool and so what I want to do here is I

want to add an AI message to my history

which uh I shouldn't repeat this but I

am actually no I'm not repeating it I'm

taking the AI response and I'm just

putting out the content let me print out

those messages again and you can see

here that it adds uh the capital Francis

Paris to the end of my chat history

which makes it easy for me to work with

and another cool functionality of this

too is link chain makes it extremely

simple to save this chat history so you

can go ahead and load it later a lot of

really cool functionality I encourage

you to go check out the next concept

we're going to look at is chains so in

this case we're going to be combining

different llm calls and actions

automatically so say you have one input

but then the output of that language

model you want to use as the input to

another call and then another call and

then another call well in that case

you're going to be using chains which is

where the chain and Lang chain comes

from so in this case we're going to

cover two of them there's a lot of

really complicated examples here I

encourage you again to go check out the

documentation to see if one of them

would cover your use case better than

what you're seeing here the first one is

going to be a simple sequential chain

and in this case I'm going to go ahead

and tell it hey I want you to do X and

then Y and then Z now the reason why

this is important

or why I like to do it is because it

helps break up the tasks Now language

models can get distracted sometimes and

if you ask it to do too many things in a

row it could get it could get confused

it could start to hallucinate and that's

not good for anybody Plus

I want to make sure that my thinking is

sound and that way I can kind of check

out the different outputs of each one of

my different actions here so in this

case I'm going to import the simple

sequential chain let me go ahead and run

this and I'm going to put two different

things to here I'm going to use two

different prompt templates so your job

is to come up with a classic dish from

the area that the user suggests I'm

going to input the user location and I'm

going to give it the user location which

we'll we'll do in a second here

and I'm going to create a llm chain with

this and I'm going to call it location

chain which basically is going to take

my language model it's going to take a

prompt template okay

and then the next one we're going to

look at given a meal give a short and

simple recipe on how to make that dish

at home so in this case we have the user

location which that's not actually what

we want we want user meal output this

wouldn't have mattered because I had the

variables the same but it just to make

it more clear

given a meal okay cool your response I'm

going to do the same thing I'm going to

put that into a meal chain so what it's

going to do is it's going to Output a

meal a classic dish and then it's going

to Output a simple recipe for that

classic dish

okay I'm gonna create my simple

sequential chain and in this case I'm

going to specify My Chains as my

location chain and then the meal chain

order matters be careful on that I'm

going to set verbose equals true which

means that it's going to tell us what

it's thinking and it's actually going to

print those statements out so it's

easier to debug what's going on

let's go ahead and create that and then

I'm going to say My overall chain I want

you to run and in this case I only have

one input variable which is going to be

Rome which is going to be the user

location that I start in the first place

let me go ahead and run this so you can

see here that it's entering the new

sequential chain and it ran Rome against

the first prompt template and got me a

classic dish which is really cool

and then it gave me a recipe to on how

to make that classic dish which is

really cool so all of a sudden it just

did two different runs for me all in one

go and I didn't have to run any

complicated code I could just use

langtain for that it's pretty sweet

now the next one that I want to show is

one that I use quite often which is

going to be the summarization chain the

reason why this one was so cool is

because if you have a long piece of text

and you want it summarized or say you

have an article you want summarized or a

tweet thread or a Hacker News Post or

whatever it may be you're going to want

to Chunk Up Your longer piece of text

and you're going to want to find you're

going to want to find summaries of those

different chunks and then get a final

summary and in that case what we're

going to do is we're going to load in

load summarize chain and we're going to

do Paul Graham's essay disk not even

sure what that one's about then we're

going to split it up into different

texts right here the chunk size is going

to be 700 and then I'm going to load

summarize chain and the chain type that

I'm going to do is going to be that one

that I mentioned beforehand which is

where you get the small summaries of the

individual sections and then you get a

summary of the small summaries I have a

whole video on different chain types and

so if you're curious go check out the

video up above and you can go see it

let me go ahead and run this and so as

you can see here the language model is

asking I'm sorry the chain or Lang chain

is asking the language model to

summarize this piece of text right here

and then this piece of text right here

because we only had two chunks that we

wanted to summarize and then it's asking

for a final concise summary so here's

the summary of the chunk number one

here's the summary of Chunk number two

and it's asking for a summary of the

summaries and we finally get a summary

of the summaries which is really cool

because all built into this one liner

right here was all the different calls

back and forth to figure out how to do

the summary of the summaries which is

one of the powers of Lane chain which is

really sweet the last thing we're going

to look at is agents and this is one of

the most complicated Concepts within

link chain which is why we're talking

about it last here but I thought that

the official link chain documentation

did a great job describing what agents

are

some applications will not require just

a predetermined chain of calls to llms

and other tools what we did up above was

a predetermined chain here but

potentially an unknown chain that

depends on the user input an unknown

chain emphasis mine means that we're not

really sure what route we want to take

but we want the language model to tell

us which route it thinks that it should

take

in these types of chains there is an

agent which has an access to a suite of

tools depending on the user input the

agent can then decide which if any of

these tools to call

so for example hey you have two

databases you could pick information

from they're on completely different

topics the user just asked you a

question about uh trees

which database should you go looking to

find your tree information well an agent

can decide that which is really sweet so

I'm going to go over the vocabulary

first and then we're going to look at an

example so an agent is the language

model that is going to be driving the

decision making cool

tools or tool is going to be a

capability of the agent so you can think

of this as similar to the open AI

plugins that just came out you can also

think of this as the ability to go

search Google the ability to go lick

your email whatever it may be

a tool kit is going to be a collection

of tools so an agent will have a toolkit

of tools uh

an agent will have a toolkit of tools

and that's what that's what it's going

to do there I'm going to import load

tools I'm going to initialize the agent

I'm going to import openai as well

with that I'm going to create my

language model now I've made I've insert

my serp API key because that's the

example that we're going to be running

through here which is an easy way to

search Google

and then with the toolkit I'm going to

go ahead and load the tools now in this

case I'm only loading one tool and it's

the server API however you could load in

a lot of tools here and you may

naturally think well let me just load it

up with all the tools in the world you

could it's just going to get difficult

for the model or the agent to know which

tool to use at which time so you kind of

only want to use the ones that you know

you're going to um

be needing at that at that point so I'm

going to pass in my language model and

I'm going to pass in my serve API here

API key then I'm going to create my

agent so I'm going to pass in the

toolkit that I just made I'm going to

pass in the language model again I'm

going to say what type of agent are you

now there's different agent types for

different types of tasks and I encourage

you to go check out the language or the

documentation to see which would be best

for you I'm going to say verbose equals

true so we can see it thinking I'm also

going to return the intermediate steps

which just means that we get more

granularity into what it's actually

doing

with this I'm going to say response uh

oh agent is not defined then what I'm

going to do here is I'm going to pass in

my query to the agent itself so what was

the first album of the band that Natalie

Bergman is a part of the reason why I

asked this question specifically is

because keep in mind I haven't uploaded

any documents here so there's no

information pre-loaded and it's kind of

a complicated question that has multiple

steps that need to be answered for it

this is a perfect question for an agent

here so let's go ahead and run this and

let's see how the agent is thinking

about it entering the new agent executor

class and it said I should try to find

out what band Natalie Bergman is a part

of so it needs to it knows that it needs

to go search which it has a Search tool

up above which I gave it and it's saying

Natalie Bergman banded so it's searching

for that one and it says observation

which is what it observed from its

action Natalie Bergman is an American

singer-songwriter she has one half the

duo of wild Belle okay cool I should

search for the debut album of wild Belle

it understood the band that she's a part

of and now it now it knows and needs to

go search for that band so it's going to

search again it's going to say wild

Belle debut album and it observes that

the debut album is Isles I know the

final answer which is good we want it to

know the finance

uh Isles is the debut album of wild

Belle the band that Natalie Bergman is a

part of that is really cool because that

is a multi-step question and uh the

agent knew what it needed to go find out

without me telling it the chain so this

chain could have been a whole lot longer

if it needed more steps with it but uh

it dynamically figured that out along

the way which is really really cool and

so if we were to print out the

intermediate steps you get more

information about what it actually did

and how it searched and all that good

information from there.



