from .agents.leader.simple_team import process_single_topic
from .agents.workers.topic_explorer import create_topic_explorer
from datetime import datetime


async def run_simple_agent_system(document_url: str):
    print(f"🔍 Starting comprehensive document processing: {document_url}")
    
    # Generate timestamp once for this entire processing session
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"📅 Processing session timestamp: {timestamp}")
    
    # Step 1: Extract topics using topic explorer
    print(f"📋 Extracting topics from document...")
    topic_explorer = create_topic_explorer()
    
    topic_prompt = f"""
    Analyze this insurance document and extract potential decision tree topics: {document_url}
    
    Make 4-5 targeted queries to comprehensively analyze:
    1. Query for eligibility criteria and requirements 
    2. Query for claim processes and approval workflows
    3. Query for coverage scenarios and benefit calculations  
    4. Query for exclusions, limitations, and conditional policies
    5. Query for risk assessment and decision-making procedures
    
    Extract 6-10 decision tree topics and provide structured output.
    """
    
    topic_result = await topic_explorer.arun(topic_prompt)
    topic_result = topic_result.content
    
    print(f"✅ Found {topic_result.total_topics_found} topics:")
    for topic in topic_result.topics:
        print(f"  • {topic.topic_name} ({topic.decision_complexity})")
    
    # Step 2: Process each topic individually using the team with shared timestamp
    print(f"\n🌳 Creating decision trees for each topic...")
    
    results = []
    for i, topic in enumerate(topic_result.topics, 1):
        print(f"  [{i}/{len(topic_result.topics)}] Processing: {topic.topic_name}")
        
        try:
            tree_result = await process_single_topic(document_url, topic.topic_name, timestamp)
            results.append({
                "topic": topic.topic_name,
                "success": True,
                "result": tree_result.content
            })
            print(f"    ✅ Completed")
        except Exception as e:
            results.append({
                "topic": topic.topic_name,
                "success": False,
                "error": str(e)
            })
            print(f"    ❌ Failed: {str(e)}")
    
    successful_trees = sum(1 for r in results if r["success"])
    
    return {
        "success": True,
        "topics_extracted": topic_result,
        "tree_results": results,
        "document_url": document_url,
        "total_topics": len(topic_result.topics),
        "successful_trees": successful_trees,
        "timestamp": timestamp
    }