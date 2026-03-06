#!/usr/bin/env python3
"""Test script to verify OpenRouter API connectivity and free model access."""

import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "qwen/qwen3-235b-a22b-thinking"

async def test_openrouter():
    """Test OpenRouter API with the chosen free model."""
    if not OPENROUTER_API_KEY:
        print("❌ No OPENROUTER_API_KEY found in .env file")
        print("💡 Get a free API key from: https://openrouter.ai/keys")
        return False
    
    print(f"🔑 Testing OpenRouter with model: {OPENROUTER_MODEL}")
    print(f"🔒 API Key: {OPENROUTER_API_KEY[:10]}...")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nikdotcrm/pascribe-bot",  # Required for free tier
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a test assistant. Keep responses very short."
            },
            {
                "role": "user",
                "content": "Say 'OpenRouter connection successful!' and briefly explain what you can help with."
            }
        ],
        "temperature": 0.3,
        "max_tokens": 100,  # Keep test response short
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", 
                                  headers=headers, json=payload) as response:
                
                print(f"📡 Response status: {response.status}")
                
                if response.status == 401:
                    print("❌ Authentication failed - check your API key")
                    return False
                elif response.status == 429:
                    print("⏰ Rate limit hit - try again later")
                    return False
                elif response.status >= 400:
                    error_text = await response.text()
                    print(f"❌ API error {response.status}: {error_text}")
                    return False
                
                result = await response.json()
                
                if 'choices' in result and result['choices']:
                    message = result['choices'][0]['message']['content']
                    print(f"✅ Success! Model response:")
                    print(f"💬 {message}")
                    
                    # Check usage info if available
                    if 'usage' in result:
                        usage = result['usage']
                        print(f"📊 Token usage: {usage.get('total_tokens', 'unknown')} total")
                    
                    return True
                else:
                    print(f"❌ Unexpected response format: {result}")
                    return False
                    
    except asyncio.TimeoutError:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    return False

async def main():
    """Main test function."""
    print("🚀 Testing OpenRouter API for Benjamin transcript analysis")
    print("=" * 60)
    
    success = await test_openrouter()
    
    print("=" * 60)
    if success:
        print("✅ OpenRouter test completed successfully!")
        print("🎯 Ready for transcript analysis")
    else:
        print("❌ OpenRouter test failed")
        print("💡 Check your API key and network connection")
        print("🔗 Get API key: https://openrouter.ai/keys")

if __name__ == "__main__":
    asyncio.run(main())