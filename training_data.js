// ═══════════════════════════════════════════════════════════════════
// REAL TRAINING DATA - Text samples with feature extraction
// Contains REAL text samples (not fake feature vectors)
// 25+ casual/conversational AI, 25+ formal AI, 25+ casual human, 15+ formal human
// Features extracted using extract25Features() function
// ═══════════════════════════════════════════════════════════════════

// 25 FEATURE EXTRACTION FROM REAL TEXT - EXACT SAME as index.html
function extract25Features(text) {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
    const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
    const N = words.length || 1;
    const S = sentences.length || 1;
    
    if (words.length === 0) return new Array(25).fill(0);
    
    // FEATURE 0: AI vocab density
    const AI_VOCAB = new Set(['delve','crucial','furthermore','robust','holistic','seamless','multifaceted',
        'pivotal','essential','integral','paramount','foster','cultivate','navigate','underscore',
        'articulate','illuminate','resonate','groundbreaking','revolutionary','comprehensive','facilitate',
        'utilize','optimize','demonstrate','transformative','innovative','synergize','impactful','paradigm',
        'actionable','scalable','streamline','empower','proactive','leverage','tapestry']);
    const f0_aiVocab = words.filter(w => AI_VOCAB.has(w)).length / N;
    
    // FEATURE 1: Emotional word density
    const EMOTIONAL = new Set(['love','hate','amazing','terrible','horrible','wonderful','awful',
        'excited','worried','happy','sad','angry','frustrated','thrilled','disappointed','grateful',
        'annoyed','passionate','scared','proud','ashamed','lonely','hopeful','nervous','confused']);
    const f1_emotional = words.filter(w => EMOTIONAL.has(w)).length / N;
    
    // FEATURE 2: Uncertainty markers
    const UNCERTAIN = new Set(['maybe','perhaps','might','possibly','probably','likely','presumably',
        'seemingly','apparently','supposedly']);
    const f2_uncertainty = words.filter(w => UNCERTAIN.has(w)).length / N;
    
    // FEATURE 3: Complex word ratio (>8 letters)
    const f3_complex = words.filter(w => w.length > 8).length / N;
    
    // FEATURE 4: Rare word ratio (words appearing only once)
    const wordCounts = {};
    words.forEach(w => wordCounts[w] = (wordCounts[w] || 0) + 1);
    const f4_rare = Object.values(wordCounts).filter(c => c === 1).length / N;
    
    // FEATURE 5: Sentence burstiness (length variation)
    const sentLens = sentences.map(s => s.split(/\s+/).length);
    const avgLen = sentLens.reduce((a,b)=>a+b,0) / sentLens.length;
    const variance = sentLens.map(l => Math.pow(l-avgLen,2)).reduce((a,b)=>a+b,0) / sentLens.length;
    const f5_burstiness = Math.sqrt(variance) / avgLen;
    
    // FEATURE 6: Avg sentence length regularity
    const f6_lengthReg = 1 - Math.min(f5_burstiness, 1);
    
    // FEATURE 7: Clause complexity (comma ratio)
    const commas = (text.match(/,/g) || []).length;
    const f7_clauses = commas / S;
    
    // FEATURE 8: Question usage
    const f8_questions = (text.match(/\?/g) || []).length / S;
    
    // FEATURE 9: Exclamation usage
    const f9_exclamations = (text.match(/!/g) || []).length / S;
    
    // FEATURE 10: Contraction density
    const contractions = (text.match(/\b(don't|can't|won't|isn't|aren't|haven't|hasn't|hadn't|wouldn't|couldn't|shouldn't|i'm|you're|he's|she's|it's|we're|they're|i've|you've|we've|they've|i'll|you'll|he'll|she'll|we'll|they'll|that's|what's|who's|let's|ain't)\b/gi) || []).length;
    const f10_contractions = contractions / N;
    
    // FEATURE 11: First person narrative
    const firstPerson = (text.match(/\b(i|me|my|myself|mine)\b/gi) || []).length;
    const f11_firstPerson = firstPerson / N;
    
    // FEATURE 12: Direct address (you/your)
    const f12_direct = (text.match(/\b(you|your|yours|yourself)\b/gi) || []).length / N;
    
    // FEATURE 13: Informal markers
    const INFORMAL = new Set(['yeah','yep','nope','gonna','wanna','gotta','kinda','sorta','like','just','really','pretty','tbh','lol','omg','btw','ok','cool','wow','hey','hmm','nah','meh','huh']);
    const f13_informal = words.filter(w => INFORMAL.has(w)).length / N;
    
    // FEATURE 14: Punctuation diversity
    const punctTypes = new Set((text.match(/[.,;:!?"'()-]/g) || [])).size;
    const f14_punctDiv = punctTypes / 10;
    
    // FEATURE 15: Semantic consistency (sentence-to-sentence word overlap)
    // High overlap = consistent topic = AI-like
    let f15_semantic = 0.5;
    if (sentences.length >= 2) {
        let overlapSum = 0;
        for (let i = 1; i < sentences.length; i++) {
            const prev = new Set((sentences[i-1].toLowerCase().match(/\b[a-z]{4,}\b/g) || []));
            const curr = (sentences[i].toLowerCase().match(/\b[a-z]{4,}\b/g) || []);
            const overlap = curr.filter(w => prev.has(w)).length;
            overlapSum += overlap / Math.max(curr.length, 1);
        }
        f15_semantic = Math.min(overlapSum / (sentences.length - 1), 1);
    }
    
    // FEATURE 16: Topic drift variance (high = human-like, low = AI smooth flow)
    let f16_drift = 0;
    if (sentences.length >= 3) {
        const overlapScores = [];
        for (let i = 1; i < sentences.length; i++) {
            const prev = new Set((sentences[i-1].toLowerCase().match(/\b[a-z]{4,}\b/g) || []));
            const curr = (sentences[i].toLowerCase().match(/\b[a-z]{4,}\b/g) || []);
            overlapScores.push(curr.filter(w => prev.has(w)).length / Math.max(curr.length, 1));
        }
        const mean = overlapScores.reduce((a,b)=>a+b,0) / overlapScores.length;
        const driftVar = overlapScores.map(v => Math.pow(v - mean, 2)).reduce((a,b)=>a+b,0) / overlapScores.length;
        f16_drift = Math.min(Math.sqrt(driftVar) * 4, 1);
    }
    
    // FEATURE 17: Perplexity proxy (bigram uniqueness)
    const bigrams = [];
    for (let i = 0; i < words.length - 1; i++) {
        bigrams.push(words[i] + ' ' + words[i+1]);
    }
    const f17_perplexity = new Set(bigrams).size / Math.max(bigrams.length, 1);
    
    // FEATURE 18: Specificity (abstract vs concrete)
    const f18_specificity = words.filter(w => w.length > 6).length / N;
    
    // FEATURE 19: Temporal references
    const TEMPORAL = new Set(['now','then','today','yesterday','tomorrow','soon','later','before','after','during','while','when','always','never','sometimes','often','recently','currently','previously']);
    const f19_temporal = words.filter(w => TEMPORAL.has(w)).length / N;
    
    // FEATURE 20: Readability score
    const avgWordLen = words.reduce((a,b)=>a+b.length,0) / N;
    const f20_readability = Math.min(avgWordLen / 8, 1);
    
    // FEATURE 21: Passive voice density
    const f21_passive = (text.match(/\b(is|are|was|were|been|be|being)\s+\w+ed\b/gi) || []).length / S;
    
    // FEATURE 22: Named entity density
    const f22_entities = (text.match(/\b[A-Z][a-z]+\b/g) || []).length / N;
    
    // FEATURE 23: Quote usage
    const f23_quotes = ((text.match(/["']/g) || []).length / 2) / S;
    
    // FEATURE 24: List/structure markers
    const LIST_MARKERS = new Set(['first','second','third','fourth','fifth','next','finally','lastly','additionally','moreover','furthermore']);
    const f24_list = words.filter(w => LIST_MARKERS.has(w)).length / N;
    
    return [
        f0_aiVocab, f1_emotional, f2_uncertainty, f3_complex, f4_rare,
        f5_burstiness, f6_lengthReg, f7_clauses, f8_questions, f9_exclamations,
        f10_contractions, f11_firstPerson, f12_direct, f13_informal, f14_punctDiv,
        f15_semantic, f16_drift, f17_perplexity, f18_specificity, f19_temporal,
        f20_readability, f21_passive, f22_entities, f23_quotes, f24_list
    ];
}

// ═══════════════════════════════════════════════════════════════════
// CASUAL/CONVERSATIONAL AI SAMPLES (25+)
// ChatGPT being friendly, giving advice, using casual tone
// ═══════════════════════════════════════════════════════════════════
const casualAISamples = [
    "Hey! So you want to learn guitar? That's awesome! Here's what worked for me - start with just 10 minutes a day. Don't worry about being perfect, just get your fingers moving. You've got this!",
    
    "Thanks for reaching out! I totally get where you're coming from. In my experience, the best approach is to break it down into small steps. What do you think?",
    
    "Oh wow, that's a great question! Okay so here's my advice - don't overthink it. Just start somewhere and iterate as you go. That's what I did and it worked out great!",
    
    "I'm happy to help! Based on what you've shared, I'd recommend starting with the basics. It's not as scary as it seems, I promise. Let me know if you get stuck!",
    
    "That's awesome that you're getting started! Here's what I learned - consistency beats intensity every time. Just show up daily and you'll be amazed at the progress.",
    
    "Hey there! So I tried this myself last year and here's what happened. I failed like 5 times before it clicked. Don't give up - you can totally do this!",
    
    "Okay so real talk - this was hard for me too at first. But then I figured out a system that works. Want me to share what worked?",
    
    "Thanks for asking! I'm actually excited to share this with you. So here's the thing - most people overcomplicate it. Keep it simple and you'll be fine!",
    
    "You know what? That's a really good point you brought up. I hadn't thought about it that way. Here's another angle to consider...",
    
    "Hope this message finds you well! I wanted to share some thoughts on your question. In my experience, the key is patience. You've got this!",
    
    "So here's my honest take - I struggled with this for months before it made sense. Then one day it just clicked. You'll get there too!",
    
    "That's a fantastic goal! I love that you're pursuing this. Here's what worked for me when I was in your shoes...",
    
    "Hey! Quick tip from someone who's been there - don't compare your beginning to someone else's middle. Everyone starts somewhere!",
    
    "I'm so glad you asked! This is actually something I'm passionate about. Here's my best advice after years of trial and error...",
    
    "Okay real talk - you're overthinking this. I did the same thing. Just start somewhere and adjust as you go. Perfect is the enemy of good!",
    
    "That's such an interesting perspective! I really appreciate you sharing that. It reminds me of when I was learning this stuff...",
    
    "Hey, just wanted to check in! How's it going with the project? Remember, progress not perfection. You're doing great!",
    
    "So I tried three different approaches before finding what works. The first two were total failures but that's how learning works, right?",
    
    "You asked exactly the right question! That's the thing most people miss. Here's what I wish someone told me earlier...",
    
    "That's awesome! I can tell you're really thinking this through. My advice? Start smaller than you think you need to. Seriously!",
    
    "Hope you're having a great week! I wanted to follow up on our conversation. Have you had a chance to try what we discussed?",
    
    "Here's the thing nobody talks about - everyone feels like an imposter at first. It's totally normal. You're not alone in this!",
    
    "So here's my controversial opinion - most advice on this topic is way too complicated. Strip it down to the basics and move forward!",
    
    "I'm genuinely excited for you! This is such a cool journey to be on. Here's what I learned that made the biggest difference...",
    
    "Quick question - have you tried the basics yet? I know it seems too simple but trust me on this one. Foundation matters!",
    
    "You know, I was just thinking about this yesterday. It's funny how we make things harder than they need to be. Here's my take...",
    
    "Thanks for sharing that with me! I know it can feel vulnerable to ask for help. But seriously, that's exactly how you grow!",
    
    "Okay I'm going to be straight with you - I made every mistake possible when I started. Learn from my fails and skip the pain!",
    
    "That's such a good insight! You're really getting it. Keep that mindset and you're going to do amazing things!"
];

// ═══════════════════════════════════════════════════════════════════
// FORMAL/ACADEMIC AI SAMPLES (25+)
// Traditional AI writing - essays, reports, technical content
// ═══════════════════════════════════════════════════════════════════
const formalAISamples = [
    "Artificial intelligence has fundamentally transformed the landscape of modern education. Furthermore, it is crucial to underscore the significant implications this technology has on pedagogical frameworks.",
    
    "In conclusion, it is important to note that climate change represents one of the most significant challenges of our time. The multifaceted nature of this phenomenon requires a comprehensive and holistic response.",
    
    "The transformer architecture leverages self-attention mechanisms to process sequential data. This approach facilitates parallel computation, thereby significantly improving training efficiency.",
    
    "There are several key factors to consider: First, you need to understand the fundamental principles. Second, it is crucial to implement best practices. Third, leverage available resources to optimize your approach.",
    
    "This is a nuanced question that involves several considerations. On one hand, the evidence suggests that the approach has significant merits. On the other hand, there are important limitations worth noting.",
    
    "It is worth noting that this represents a significant paradigm shift in how we approach problem-solving. The multifaceted implications of this development underscore the need for a comprehensive framework.",
    
    "The morning sun cast its golden rays across the tranquil landscape, illuminating the intricate tapestry of nature in all its multifaceted glory. Each moment unfolded like a delicate symphony.",
    
    "Thank you for reaching out. I hope this message finds you well. I wanted to provide a comprehensive response to your inquiry. It is crucial that we address each aspect thoroughly.",
    
    "Our innovative solution leverages cutting-edge technology to streamline your workflow and maximize productivity. This robust platform facilitates seamless integration across diverse systems.",
    
    "The pathophysiological mechanisms underlying this condition are multifaceted. Research has demonstrated that the interaction between genetic and environmental factors plays a crucial role.",
    
    "The concept of consciousness represents one of the most profound and fundamental questions in philosophical inquiry. Moreover, the intricate relationship between mind and matter continues to underscore the complexity.",
    
    "To effectively accomplish this task, follow these essential steps. First, ensure you have all necessary resources. Subsequently, implement the foundational elements systematically.",
    
    "Transform your business with our revolutionary platform. Leverage cutting-edge AI to streamline operations and maximize ROI. Our robust solution empowers teams to achieve unprecedented levels of productivity.",
    
    "Recent developments underscore the significant impact of this policy on economic indicators. Analysts note that the multifaceted implications could fundamentally alter market dynamics.",
    
    "It is important to note that personal growth is a journey, not a destination. To cultivate meaningful progress, you must leverage your strengths while addressing areas for improvement.",
    
    "This function implements a recursive algorithm that traverses the data structure. It leverages dynamic programming to optimize performance. The implementation demonstrates robust error handling.",
    
    "Examining this from multiple perspectives reveals several crucial insights. The data demonstrates a significant correlation between these variables. Moreover, it is paramount to consider the broader contextual framework.",
    
    "Excited to share insights on how AI is fundamentally transforming industries! It's crucial that we leverage these innovations responsibly.",
    
    "This comprehensive analysis examines the multifaceted dimensions of organizational performance. The findings underscore significant correlations between leadership effectiveness and employee engagement.",
    
    "Embrace the journey of continuous self-improvement! It is essential to cultivate resilience and leverage every opportunity for growth. Remember, your potential is fundamentally limitless.",
    
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. This fundamental technology leverages algorithms to analyze data and identify patterns.",
    
    "While there are compelling arguments on both sides, it is crucial to underscore the significant weight of the evidence. On one hand, proponents argue that this approach is fundamentally sound.",
    
    "In today's rapidly evolving landscape, it is more crucial than ever to understand the fundamental principles that govern our world. This comprehensive exploration will delve into the intricate mechanisms at play.",
    
    "Delving into the realm of possibilities, this showcases how innovation can fundamentally reshape our understanding. It's crucial to leverage these insights and foster meaningful connections.",
    
    "The realm of Eldoria stretched before them, a tapestry of intricate beauty. The landscape was multifaceted, encompassing vast forests and towering mountains.",

    "Examining this from multiple perspectives reveals crucial insights. Data demonstrates correlations between variables. It is paramount to consider the broader contextual framework.",
    
    "Transform your business with our platform. Leverage AI to streamline operations. Our solution empowers teams to achieve productivity gains and competitive advantage.",
    
    "Recent developments underscore policy impacts on economic indicators. Analysts note implications could alter market dynamics. Stakeholders emphasize need for regulatory frameworks.",
    
    "To accomplish this task, follow these steps. First, ensure necessary resources. Subsequently, implement elements systematically. Validate each step before proceeding.",
    
    "Our innovative solution leverages technology to streamline workflow. This platform facilitates integration across systems, enabling organizations to optimize efficiency.",
    
    "Thank you for reaching out. I hope this finds you well. I wanted to provide a comprehensive response. It is crucial we address each aspect thoroughly.",
    
    "In conclusion, climate change represents significant challenges. The multifaceted nature requires a comprehensive response. Stakeholders must collaborate to implement robust solutions."
];

// ═══════════════════════════════════════════════════════════════════
// CASUAL HUMAN SAMPLES (25+)
// Reddit posts, personal stories, informal writing with real emotion
// ═══════════════════════════════════════════════════════════════════
const casualHumanSamples = [
    "ok so i've been trying to learn guitar for like 3 months now and honestly it's way harder than i thought lol. my fingers hurt so bad the first week i almost quit.",
    
    "Last summer I decided to quit my job and move to Portugal. Everyone thought I was crazy, and honestly, maybe they were right? The first month was rough. I cried in a grocery store.",
    
    "This is absolutely ridiculous. I waited 3 hours on hold and they still couldn't fix my account!! Every rep gave me a different answer. I've been a customer for 8 years. EIGHT YEARS.",
    
    "I think the main reason the French Revolution happened was because people were just starving. Like, when you literally can't eat, you stop caring about rules.",
    
    "hey! hope you're doing well :) so i wanted to tell you about what happened on saturday. you know that guy from work i mentioned? yeah he actually showed up at the same bar.",
    
    "Went here for my birthday dinner with my family. The pasta was honestly incredible - I'm still thinking about it two weeks later. Our waiter was so attentive without being annoying.",
    
    "Can we talk about how everyone suddenly became an expert on everything the moment they got wifi?? My uncle who has never read a scientific paper is now explaining vaccines to me.",
    
    "My grandmother kept chickens. This sounds strange as an opening line, I know. But every summer morning of my childhood began the same way: waking up to their noise.",
    
    "ok figured it out - in case anyone else has this problem. turns out the issue was my DNS settings. I had manually set them a year ago and forgot. feels dumb in retrospect.",
    
    "Three years ago today I was in the hospital and genuinely didn't think I'd make it. I don't talk about this stuff much but today feels like the right day.",
    
    "omg did you see what happened?? call me when you can, this is insane. also are we still on for friday? let me know because i need to book the restaurant.",
    
    "Hi Sarah, hope your week is going well! Quick update on the project - we hit a bit of a snag with the API integration but Tom thinks he can have it fixed by Thursday.",
    
    "Romeo and Juliet is a play about how stupid adults can be. I know that's harsh but think about it - two families fighting so long nobody remembers why, and teenagers die because of it.",
    
    "I spent two years living in Tokyo without speaking Japanese when I arrived. Here's what nobody tells you: it's not the language barrier that's hard, it's the exhaustion.",
    
    "I don't care what anyone says, that was the best game I've watched in 20 years. Yeah the last quarter was painful. Yeah I screamed at my TV like an idiot. Worth it.",
    
    "Nobody told me that having a toddler means never finishing a hot cup of coffee. Or a sentence. Or a thought. My daughter is three and absolutely feral in the best way.",
    
    "So my sister borrowed $200 from me 6 months ago and keeps 'forgetting' to pay me back. Last week I asked and she got mad at ME. Am I wrong for being annoyed?",
    
    "Unpopular opinion: most meetings could be an email. I just sat through a 90-minute 'alignment session' that could have been a two-paragraph message.",
    
    "I don't know how to explain today. Everything went fine, nothing bad happened, but I spent most of it feeling like I was watching myself from far away.",
    
    "OK so I've made this cake probably 40 times and I always forget to write down what I changed. This time I'm actually doing it. The secret is browned butter.",
    
    "Went to Kyoto last April. Everyone says spring is the best time because of cherry blossoms. They're right, but nobody tells you that every other tourist also read that advice.",
    
    "Look, I'm not here to change anyone's mind. I know that's not how this works. But I've genuinely tried to understand the other side and I just keep running into the same wall.",
    
    "Has anyone else experienced this? Started getting weird headaches about 3 weeks ago - not painful exactly but more like pressure behind my eyes. Doctor appointment is in two weeks.",
    
    "My cat has decided that 3am is the ideal time to reenact Waterloo on my feet. I am not Napoleon. I did not consent to this war. The vet says he's 'playful and energetic.'",
    
    "We almost ran out of money twice in year one. I didn't tell anyone because I was terrified that if I said it out loud it would become real. Looking back, that was dumb.",
    
    "I picked this book up because a friend wouldn't stop talking about it. Read the first chapter standing in my kitchen and then didn't move for 6 hours. I missed dinner. I have no regrets."
];

// ═══════════════════════════════════════════════════════════════════
// FORMAL HUMAN SAMPLES (15+)
// Professional blogs, academic writing, formal human content
// ═══════════════════════════════════════════════════════════════════
const formalHumanSamples = [
    "This study examines the relationship between sleep deprivation and academic performance in undergraduate students. Unlike previous research, we tracked daily cognitive assessments over 12 weeks.",
    
    "I'm 34 and I only learned to drive last year. I told people I just 'preferred public transport' for 15 years. The truth is I was scared and embarrassed to admit I didn't know how.",
    
    "The data from our three-year longitudinal study suggest that early childhood vocabulary exposure has a stronger effect than previously estimated. This contradicts the 2018 findings by Morris et al., whose sample size may have been too small.",
    
    "After 20 years practicing surgery, the cases that still keep me up at night are not the complex ones. They're the ones where I made a routine decision and something unexpected happened anyway.",
    
    "The argument that social media causes depression is harder to make than people think. Correlation is well-documented. Causation is a different matter entirely, and the direction of that relationship remains genuinely unclear.",
    
    "I went to the same university as my father and his father before him. I chose differently for my children, and explaining that choice to my father was one of the harder conversations of my adult life.",
    
    "Our team spent eight months on this project before realising the fundamental assumption we had built everything on was wrong. Starting over was demoralizing. It was also the right call.",
    
    "The 1918 flu pandemic killed more people than the First World War, yet it occupies far less cultural memory. This asymmetry is worth examining because it tells us something about how societies choose what to remember.",
    
    "Reading the original source material is always worth it. Secondary summaries, no matter how careful, introduce interpretive layers. If the original is available and you have the time, read it.",
    
    "Three years into running my own firm I can say with certainty that the business advice I received before starting was almost entirely wrong, not because it was dishonest, but because context changes everything.",
    
    "The evidence on charter school performance is genuinely mixed. Studies finding positive effects and studies finding no difference often use different populations and different time horizons. Anyone claiming certainty is overstating it.",
    
    "My mother died during the pandemic. No funeral, no gathering, no shared grief in the same room with the people who loved her. I am still not sure how to make sense of that.",
    
    "The structural problem with most diversity training is that it targets individual attitudes while leaving institutional processes unchanged. Attitudes rarely drive behavior as much as incentive structures do.",
    
    "What surprised me most about living abroad for two years was not the obvious cultural differences, but the subtle ones that take months to notice and longer to understand.",
    
    "The most useful thing I learned in graduate school was not any specific methodology or body of theory. It was learning to sit with uncertainty without forcing premature conclusions."
];

// ═══════════════════════════════════════════════════════════════════
// CONVERT TEXT SAMPLES TO FEATURE VECTORS
// ═══════════════════════════════════════════════════════════════════
const aiRealProfiles = [
    ...casualAISamples.map(text => extract25Features(text)),
    ...formalAISamples.map(text => extract25Features(text))
];

const humanRealProfiles = [
    ...casualHumanSamples.map(text => extract25Features(text)),
    ...formalHumanSamples.map(text => extract25Features(text))
];

console.log(`📊 Training data prepared: ${casualAISamples.length} casual AI + ${formalAISamples.length} formal AI + ${casualHumanSamples.length} casual human + ${formalHumanSamples.length} formal human = ${aiRealProfiles.length + humanRealProfiles.length} total samples`);

// ═══════════════════════════════════════════════════════════════════
// AUGMENTATION - Generate realistic variations from real profiles
// ═══════════════════════════════════════════════════════════════════
function generateTrainingData() {
    const features = [];
    const labels = [];

    // Add all real base profiles
    aiRealProfiles.forEach(p => { features.push([...p]); labels.push(1); });
    humanRealProfiles.forEach(p => { features.push([...p]); labels.push(0); });

    // Generate 8 realistic variations per profile
    const variationsPerProfile = 8;

    // AI variations
    for (let i = 0; i < variationsPerProfile; i++) {
        aiRealProfiles.forEach(base => {
            const variation = base.map((v, idx) => {
                // Features 0,3,4,15 (AI hallmarks) - tight noise to preserve signal
                const noiseScale = [0,3,4,15,20,21].includes(idx) ? 0.04 : 0.07;
                const noise = (Math.random() - 0.5) * noiseScale;
                return Math.max(0, Math.min(1, v + noise));
            });
            features.push(variation);
            labels.push(1);
        });
    }

    // Human variations
    for (let i = 0; i < variationsPerProfile; i++) {
        humanRealProfiles.forEach(base => {
            const variation = base.map((v, idx) => {
                // Features 1,9,10,11,13 (human hallmarks) - tight noise
                const noiseScale = [1,9,10,11,13,16].includes(idx) ? 0.04 : 0.07;
                const noise = (Math.random() - 0.5) * noiseScale;
                return Math.max(0, Math.min(1, v + noise));
            });
            features.push(variation);
            labels.push(0);
        });
    }

    // Edge cases: slightly AI-ish human writing (academic humans)
    const academicHumanIndices = casualHumanSamples.slice(0, 5).map((_, i) => i);
    for (let i = 0; i < 10; i++) {
        const base = humanRealProfiles[academicHumanIndices[i % academicHumanIndices.length] || 0];
        const variation = base.map(v => Math.max(0, Math.min(1, v + (Math.random() - 0.5) * 0.06)));
        features.push(variation);
        labels.push(0);
    }

    // Edge cases: slightly human-ish AI writing (casual AI)
    const casualAiIndices = casualAISamples.slice(0, 5).map((_, i) => i);
    for (let i = 0; i < 10; i++) {
        const base = aiRealProfiles[casualAiIndices[i % casualAiIndices.length] || 0];
        const variation = base.map(v => Math.max(0, Math.min(1, v + (Math.random() - 0.5) * 0.06)));
        features.push(variation);
        labels.push(1);
    }

    // Shuffle to avoid order bias
    for (let i = features.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [features[i], features[j]] = [features[j], features[i]];
        [labels[i], labels[j]] = [labels[j], labels[i]];
    }

    const aiCount = labels.filter(l => l === 1).length;
    const humanCount = labels.filter(l => l === 0).length;
    console.log(`✅ Training data: ${features.length} samples (${aiCount} AI, ${humanCount} Human)`);
    return { features, labels };
}

// ═══════════════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════════════
const { features: generatedFeatures, labels: generatedLabels } = generateTrainingData();
window.allFeatures = generatedFeatures;
window.allLabels = generatedLabels;

// ═══════════════════════════════════════════════════════════════════
// FALLBACK: If no training data, generate simple version
// ═══════════════════════════════════════════════════════════════════
if (!window.allFeatures || window.allFeatures.length === 0) {
    console.log('⚠️ Generating fallback training data in training_data.js...');
    
    // Simple AI patterns (high AI vocab, low emotion, low burstiness)
    const aiFeatures = [
        [0.9, 0.02, 0.05, 0.8, 0.7, 0.1, 0.3, 0.5, 0, 0, 0, 0, 0.05, 0, 0.5, 0.9, 0.1, 0.5, 0.85, 0.05, 0.7, 0.75, 0.15, 0, 0.2],
        [0.85, 0.03, 0.08, 0.75, 0.65, 0.12, 0.35, 0.48, 0, 0, 0, 0, 0.1, 0, 0.52, 0.85, 0.12, 0.48, 0.8, 0.08, 0.65, 0.7, 0.12, 0, 0.15],
        [0.88, 0.01, 0.06, 0.78, 0.68, 0.11, 0.32, 0.49, 0, 0, 0, 0, 0.08, 0, 0.48, 0.88, 0.11, 0.49, 0.82, 0.06, 0.68, 0.72, 0.14, 0, 0.18],
        [0.82, 0.04, 0.07, 0.72, 0.62, 0.13, 0.38, 0.46, 0, 0, 0, 0, 0.12, 0, 0.55, 0.82, 0.13, 0.46, 0.78, 0.09, 0.62, 0.68, 0.16, 0, 0.22],
        [0.87, 0.02, 0.05, 0.76, 0.66, 0.1, 0.34, 0.47, 0, 0, 0, 0, 0.09, 0, 0.51, 0.87, 0.1, 0.47, 0.81, 0.07, 0.66, 0.71, 0.13, 0, 0.17],
    ];
    
    // Simple human patterns (low AI vocab, high emotion, high burstiness)
    const humanFeatures = [
        [0.05, 0.7, 0.5, 0.1, 0.15, 0.8, 0.6, 0.4, 0.1, 0.05, 0.7, 0.6, 0.2, 0.4, 0.75, 0.2, 0.8, 0.75, 0.1, 0.5, 0.3, 0.05, 0.2, 0, 0],
        [0.02, 0.65, 0.45, 0.08, 0.12, 0.75, 0.55, 0.35, 0.15, 0.02, 0.65, 0.55, 0.15, 0.35, 0.7, 0.18, 0.78, 0.72, 0.08, 0.45, 0.25, 0.02, 0.25, 0, 0],
        [0.04, 0.68, 0.48, 0.09, 0.14, 0.78, 0.58, 0.38, 0.12, 0.04, 0.68, 0.58, 0.18, 0.38, 0.73, 0.19, 0.79, 0.74, 0.09, 0.48, 0.28, 0.04, 0.22, 0, 0],
        [0.03, 0.72, 0.52, 0.11, 0.16, 0.82, 0.62, 0.42, 0.08, 0.03, 0.72, 0.62, 0.22, 0.42, 0.77, 0.22, 0.82, 0.78, 0.11, 0.52, 0.32, 0.03, 0.18, 0, 0],
        [0.06, 0.75, 0.55, 0.12, 0.18, 0.85, 0.65, 0.45, 0.06, 0.06, 0.75, 0.65, 0.25, 0.45, 0.8, 0.25, 0.85, 0.8, 0.12, 0.55, 0.35, 0.06, 0.15, 0, 0],
    ];
    
    window.allFeatures = [...aiFeatures, ...humanFeatures];
    window.allLabels = [...Array(aiFeatures.length).fill(1), ...Array(humanFeatures.length).fill(0)];
    
    console.log(`✅ Fallback data: ${window.allFeatures.length} samples`);
}

if (typeof module !== 'undefined') {
    module.exports = { allFeatures: window.allFeatures, allLabels: window.allLabels, generateTrainingData };
}