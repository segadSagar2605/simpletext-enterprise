# Gemini API - Setup & Billing Guide

## 1. WHAT IS GEMINI API?

Gemini API is Google's AI platform that provides:
- **Text Embeddings** (models/embedding-001) - Convert text to vectors
- **LLM Chat** (models/gemini-1.5-flash, gemini-2.0-flash) - Chat completions
- **Image Analysis** - Vision capabilities

**For our use case**, we need the **Embedding model** to convert document chunks into vectors.

---

## 2. GETTING STARTED - NO CREDIT CARD REQUIRED ✅

### Step 1: Sign Up
- Go to: https://ai.google.dev/
- Click **"Get Started"** or **"Sign Up"**
- Use your Google account (Gmail)
- No credit card needed upfront

### Step 2: Get Free API Key
- Go to: https://makersuite.google.com/app/apikey
- Click **"Create API Key"**
- It generates instantly and starts with `AIza_`
- Copy and save it

### Step 3: Check Free Quota
- Under "API Keys" section, you can see your free tier limits
- Free tier includes: **60 requests per minute** and **Daily quota**

---

## 3. PRICING - THIS IS IMPORTANT ⚠️

### Free Tier (Always Available)
```
Embedding Model (embedding-001):
- First 100 requests: FREE
- After that: $0.00001 per request

For context:
- 1 Document = ~10 chunks = 10 requests
- 1000 Documents = 10,000 requests
- Cost = 10,000 × $0.00001 = $0.10 ✅ VERY CHEAP
```

### Payment Setup
**You need to enable billing to go beyond free tier:**
1. Go to: https://console.cloud.google.com/
2. Create a "Project" (if not exists)
3. Enable Billing → Add payment method (credit/debit card)
4. Set a **monthly budget limit** to avoid surprises

### Cost Examples
```
10 Documents:    $0.0001 (negligible)
100 Documents:   $0.001 (negligible)
1,000 Documents: $0.01  (1 cent)
10,000 Documents: $0.10 (10 cents)
100,000 Documents: $1.00 (1 dollar)
```

**Annual cost for 10,000 documents**: ~$1.20

---

## 4. BILLING - HOW IT WORKS

### Scenario: You upload 50 documents

```
Timeline:
─────────────────────────────────────────

Day 1: Upload 50 documents
       └─ System processes: 50 docs × 10 chunks = 500 requests
       └─ Cost: 500 × $0.00001 = $0.005
       └─ Still in FREE tier (first 100 requests free)

Day 2: Upload 100 more documents  
       └─ System processes: 100 × 10 = 1,000 requests
       └─ Cost: 1,000 × $0.00001 = $0.01
       └─ Now you've used 1,500 total requests
       └─ You're past free tier, but cost is still $0.015 total

Monthly: Upload 1,000 documents
         └─ Total requests: 10,000
         └─ Total cost: $0.10 per month
         └─ Annual: $1.20
```

**Billing appears on your Google Cloud invoice** at the end of the month.

---

## 5. RATE LIMITS - IMPORTANT ⚠️

Free tier includes:
- **60 requests per minute** (embedding model)
- **Enough for ~6 document uploads per minute**

If you upload faster:
```
Normal flow:        FAST flow (10 docs at once):
Upload 1 doc       Upload 10 docs simultaneously
└─ 10 requests     └─ 100 requests per minute
└─ Takes: 10 sec   └─ EXCEEDS 60 req/min limit ❌
                   └─ Some requests rejected
                   └─ Need to wait or queue
```

**Solution**: We'll add request queuing in the code (upload 1 doc at a time or batch).

---

## 6. ADVANTAGES vs DISADVANTAGES

### ✅ Advantages
| Feature | Benefit |
|---------|---------|
| **No startup delay** | App launches in 1 second (no model download) |
| **Instant response** | API call takes ~500ms per batch |
| **Very cheap** | $0.10 for 10,000 documents |
| **Always available** | No local RAM/CPU needed |
| **Scales easily** | Can process 1M documents without setup changes |
| **Quality** | Google's embeddings are excellent |

### ⚠️ Disadvantages
| Issue | Impact |
|-------|--------|
| **Requires internet** | No offline mode |
| **API dependent** | If Google API is down, indexing fails |
| **Rate limited** | 60 req/min on free tier (fine for typical use) |
| **Quota tracking** | Need to monitor usage |
| **Card required** | For production (billing enabled) |

---

## 7. LOCAL MODEL vs GEMINI API COMPARISON

```
METRIC                  LOCAL              GEMINI API
────────────────────────────────────────────────────
First startup           60 seconds         1 second ✅
Subsequent startups     5 seconds          1 second ✅
RAM usage               400MB              5MB ✅
Storage                 100MB              0MB ✅
Internet required       No                 Yes ⚠️
Cost                    $0                 $0-1/month
Scalability             Poor (1-2 docs)    Excellent (1000s) ✅
Embedding quality       Medium             Excellent ✅
Offline mode            Yes                No ⚠️
Setup complexity        Medium             Simple ✅
```

---

## 8. WHEN TO USE GEMINI API

✅ **Use Gemini API if:**
- You want fast startup (skip model download)
- You're uploading <10,000 docs per month ($1 cost)
- You have internet connection available
- You want minimal local resource usage

❌ **Keep local model if:**
- You need offline mode
- You're processing 100,000+ docs/month (cost becomes noticeable)
- You want zero API dependency
- You have limited internet bandwidth

---

## 9. BILLING SETUP CHECKLIST

Before we start coding, you'll need:

- [ ] Google account (you have this)
- [ ] Go to https://makersuite.google.com/app/apikey
- [ ] Create API key (free tier, copy the key)
- [ ] For production: Set up billing at https://console.cloud.google.com/
- [ ] Add a credit/debit card
- [ ] Set monthly budget limit (e.g., $10)

---

## 10. RECOMMENDED SETUP

### For Your Use Case:

```
Setup: Gemini API for embeddings
Free Tier: 
  - 60 requests/minute
  - First 100 requests free
  - Then $0.00001 per request

Budget:
  - Documents per month: Up to 10,000
  - Monthly cost: ~$0.10
  - Annual cost: ~$1.20
  - Budget limit: Set to $10/month (safe)

Result:
  - App startup: 1 second ✅
  - Embedding speed: 500ms per batch ✅
  - No local RAM usage ✅
  - Scales to 100,000s of docs ✅
```

---

## 11. TROUBLESHOOTING PRE-EMPTION

**Q: What if I exceed my budget?**
A: Requests are rejected with error. You won't be charged beyond limit.

**Q: What if Google API is down?**
A: Indexing fails. We'd add fallback to local model in code.

**Q: Can I switch back to local model?**
A: Yes! Easy switch - 2 minutes to revert.

**Q: Do I need to manage quotas?**
A: Minimal - free tier covers typical usage automatically.

---

## NEXT STEPS - WAITING FOR YOUR GO-AHEAD

Once you confirm, we'll:
1. ✅ Get the API key from makersuite.google.com
2. ✅ Set GEMINI_API_KEY environment variable in PowerShell
3. ✅ Update indexer.py to use Gemini API
4. ✅ Update main.py search function
5. ✅ Test with a document upload
6. ✅ Verify billing shows correctly

**Questions before we proceed?**
