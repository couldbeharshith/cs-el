import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.decision_tree.simple_agent_builder import run_simple_agent_system

async def main():
    document_url = "https://micbdyubdfqefphlaouz.supabase.co/storage/v1/object/public/documents/OrientalInsure.pdf"
    
    print("🚀 Running Simple Agent Team System...")
    print(f"📄 Document: {document_url}")
    print("🤖 Using topic explorer + decision tree worker team")
    
    results = await run_simple_agent_system(document_url)

    print(f"\n📊 Processing Results:")
    print(f"  Status: {'✅ Success' if results['success'] else '❌ Failed'}")
    print(f"  Document: {results['document_url']}")
    print(f"  Topics extracted: {results['total_topics']}")
    print(f"  Decision trees created: {results['successful_trees']}")
    print(f"  Success rate: {(results['successful_trees']/results['total_topics']*100):.1f}%")
    
    print(f"\n📋 Topic Analysis:")
    topics = results['topics_extracted']
    print(f"  Analysis: {topics.analysis_summary}")
    
    print(f"\n🌳 Individual Topic Results:")
    for result in results['tree_results']:
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {result['topic']}")
        if not result["success"]:
            print(f"    Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
