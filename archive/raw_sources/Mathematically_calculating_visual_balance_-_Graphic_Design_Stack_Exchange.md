# Mathematically calculating visual balance - Graphic Design Stack Exchange
Source URL: https://graphicdesign.stackexchange.com/questions/78996/mathematically-calculating-visual-balance

Mathematically calculating visual balance
Ask Question
Asked 9 years, 7 months ago
Modified 9 years, 7 months ago
Viewed 639 times
6

Is it possible to mathematically calculate visual balance?

Below is an example of what I mean. Both images contain the letters "M" and "C," but are aligned differently. In the first image, the letters are centered and formatted to have normal kerning. In the second, the "C" is shifted to the right to improve the visual balance of the image as a whole.

Since the visual weight in the letter "C" is on the left and not centered as it is in the letter "M," the visual balance of the first example is off.

Would it be effective to mathematically determine a weighted average of the letters (white parts) and adjust spacings accordingly? This could be done by defining a function to represent the total height of the white bits at any given x value, then determining the weighted average (integral of x*f(x)) to find the visual "center" of the text and adjusting the text's position so that that "center" is in the image's center.

mathematicsbalance
Share
Improve this question
Follow
asked Oct 20, 2016 at 6:02
DefinitelyNotAPlesiosaur
3451
1 gold badge
3
3 silver badges
8
8 bronze badges
2
similar: graphicdesign.stackexchange.com/questions/57620/… and graphicdesign.stackexchange.com/questions/49527/… – 
PieBie
♦
 
Commented
Oct 20, 2016 at 9:13
Add a comment
3 Answers
Sorted by:
Highest score (default)
Date modified (newest first)
Date created (oldest first)
5

What your essentially asking is a special case of kerning. Systems that do automatic kerning have been done. This is called optical kerning in inDesign, you can find a few other companies have made their own tools. While its true that manual kerning make for better results in many cases than optical kerning there are situations where anything better than nothing is acceptable.

However, the algorithms behind different kerning solutions are usually trade secrets. But the lore knows a few patents and papers on how to automate kerning (see google scholar search for example). Overall this is a very complicated thing as the human visual system is incredibly complicated and kerning needs to account for very many things to be perfect.

So while doing a automated system may take quite a lot of development time to give even acceptable solutions, there is no doubt that such systems could not be made. In your case a custom system that may or may not pay itself back. It may be worth the investment in some special cases.

In any case if there is a formula its probably is hideously complex, and we dont know it as of yet. But that does not mean it does not exist. So currently you may be better off manually doing this in most cases that matter in any way.

Share
Improve this answer
Follow
answered Oct 20, 2016 at 17:47
joojaa
58.9k8
8 gold badges
89
89 silver badges
185
185 bronze badges
2
Notably, the context of this example of two separate (?) letters is very different from plain running text, which in turn is again different from all caps headlines. To top it off, actual size also matters. A 200 pt line on a billboard is kerned much tighter than the same text on an A5 flyer. – 
Jongware
 
Commented
Oct 20, 2016 at 17:55
Add a comment
2

typography itself has its rules to balance the types, ascendents, descendents, width, height, center, etc. balancing text is kinda about grasping the right spot to use as your reference frame for balancing. As in: if you're gonna use the center, the border, a special midpoint of the type, etc.

Share
Improve this answer
Follow
answered Oct 20, 2016 at 20:25
Micalatéia
1294
4 bronze badges
Add a comment
-1

Due to the nature of the origins of StackOverflow, this sort of desire to reduce design to algorithms based on quack-like research analysis appears from time to time.

TRENDS exist in ALL THINGS SUBJECTIVE.

From "chaos theory" onwards, scientism has posited it's only a matter of time until all evolution, ebbs, flows and events can be divined. Until that fairy tale comes true, put some faith in the ideas, innovations, intuitions and instincts of designers.

Placement, spacing, weighting, colouring, contrast, disparity, inference and all the aspects of all things that make up balance vary...

They vary based on (amongst other things) platform, target, medium, purpose, audiences, cultures, times, interests, genres, niches, messages and contexts.

Trends, by their very nature, are liquid, and evolve and revolve from (again, amongst other things) influence, inspiration, invention and innovation.

Add to this the library of experiences in the utilisation and iteration on ideas from both originality and the twins of intuition and instinct (that lay at the very heart of a great designer) and you'll begin seeing the challenge for what it is. Impossible. Insurmountable.

But please don't forget the significance and impact of juxtaposition, revolution, deliberate disruption and nuanced visual harking and symbology deliberately designed to stir memories and references.

I hope you begin to understand design is a CREATIVE ART.

So, no. It is not possible to formulate algorithms that achieve "balance".

It is highly unlikely that any sense of visual balance (for the purposes of influence) is actually relative to any mathematical models of balance. And any moment in tides and trends when there is a correlation is pure fluke.

Attempting to mimic one tiny, subset of the trends that exist in design has not been achieved, (except for things like musicians modelling themselves on others) let alone anything that considers the more profound effects of mental cognition, recognition and ignition that created imagery is specifically designed to evoke.

And there's the keys to the kingdom of design. Graphic design has purpose - to influence humans. Code can't and won't ever understand the contexts and experiences of humans and how they shape their relationships, realities and reactions to that which they see and perceive.

If all the above fails to stir you, then remember this:

Letters, themselves, are not geometric and algorithmic in nature. They're evolved, imperfect, representations that are both comprised of and compromised in their formulation, form, function and functionality.

They're also not at all universal.

Hire a designer.

ps: Oddly, I think programmers have a vastly greater storage capacity for visual relationships, but that they perhaps archive in a non visual manner. Just look at the enormity of programming languages, frameworks, APIs and libraries they hold within to use and abuse, then the intertwining nature of code in even a simple app. Stunning stuff! That's a myriad of relationships and interlinking that's simply bigger than the biggest ever infographic. And they hold this in their heads!!!

Share
Improve this answer
Follow
edited Oct 20, 2016 at 19:10
Ryan
23.1k16
16 gold badges
88
88 silver badges
158
158 bronze badges
answered Oct 20, 2016 at 8:09
Confused
2,8852
2 gold badges
17
17 silver badges
25
25 bronze badges
While I agree in most part with your answer, I think the tone is sometimes a bit harsh and condescending. Also, there is no mention of code or programmers in the question, so your conclusion is a bit 'off the bat'. – 
PieBie
♦
 
Commented
Oct 20, 2016 at 17:56
Add a comment

Start asking to get answers

Find the answer to your question by asking.

Ask question

Explore related questions

mathematicsbalance

See similar questions with these tags.

Featured on Meta
Native Ads Coming To Comments
Report this ad
Report this ad
Linked
47
Aligning letters "Wrong" appears more "Right"
11
Is there a name, term, or practice for off-centering things based on volume rather than by width and height?
Related
8
What benefit does proper balance add to a design?
4
Which tool to balance color strength ?
0
How to balance a tab bar with an even number of elements, one of them being a call-to-action?
39
What methods can I use to create balance and consistency between a group of differing logos?
2
Trying to find balance in uneven text blocks
7
I cannot figure how to visually balance this falcon logo for my school
Hot Network Questions
What is boil-off and how to stop it from happening in the vacuum of space?
A question about parallel programming (OpenMP and MPI)
крылатый meaning in context
Nowhere to be seen (striking image)
Bricked laptop won't start
more hot questions
 Question feed
By continuing to use this website, you agree Stack Exchange can store cookies on your device and disclose information in accordance with our Cookie Policy. By exiting this window, default cookies will be accepted. To reject cookies, select an option from below.
Necessary cookies only
Customize settings
Cookie consent preference center
When you visit any of our websites, it may store or retrieve information on your browser, mostly in the form of cookies. This information might be about you, your preferences, or your device and is mostly used to make the site work as you expect it to. The information does not usually directly identify you, but it can give you a more personalized experience. Because we respect your right to privacy, you can choose not to allow some types of cookies. Click on the different category headings to find out more and manage your preferences. Please note, blocking some types of cookies may impact your experience of the site and the services we are able to offer.
Cookie policy
Accept all cookies
Manage consent preferences
Strictly Necessary Cookies
Always Active

These cookies are necessary for the website to function and cannot be switched off in our systems. They are usually only set in response to actions made by you which amount to a request for services, such as setting your privacy preferences, logging in or filling in forms. You can set your browser to block or alert you about these cookies, but some parts of the site will not then work. These cookies do not store any personally identifiable information.

Targeting Cookies
 Targeting Cookies

These cookies are used to make advertising messages more relevant to you and may be set through our site by us or by our advertising partners. They may be used to build a profile of your interests and show you relevant advertising on our site or on other sites. They do not store directly personal information, but are based on uniquely identifying your browser and internet device.

Performance Cookies
 Performance Cookies

These cookies allow us to count visits and traffic sources so we can measure and improve the performance of our site. They help us to know which pages are the most and least popular and see how visitors move around the site. All information these cookies collect is aggregated and therefore anonymous. If you do not allow these cookies we will not know when you have visited our site, and will not be able to monitor its performance.

Functional Cookies
 Functional Cookies

These cookies enable the website to provide enhanced functionality and personalisation. They may be set by us or by third party providers whose services we have added to our pages. If you do not allow these cookies then some or all of these services may not function properly.

Cookie List

 
Clear
 checkbox label label
Apply Cancel
Consent Leg.Interest
 checkbox label label
 checkbox label label
 checkbox label label
Necessary cookies only Confirm my choices
Report this ad