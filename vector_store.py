# vector_store.py
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "engagement-vector")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

INPUT_JSON = os.getenv("INPUT_JSON", "engagement_data.json")

def build_text_for_embedding(item: dict, item_type: str, item_id: str) -> str:
    """
    Construct a searchable text blob from engagement data.
    Different formats for different data types.
    """
    parts = []
    
    if item_type == "multi_channel_campaigns":
        parts.append(f"Campaign Type: {item.get('campaign_type', '')}")
        parts.append(f"Description: {item.get('description', '')}")
        
        if 'best_practices' in item:
            parts.append("Best Practices:")
            for practice in item['best_practices']:
                parts.append(f"  - {practice}")
        
        if 'templates' in item:
            parts.append("Templates:")
            for template in item['templates']:
                parts.append(f"  {template.get('template_name', '')}: {template.get('subject', template.get('message', ''))}")
                if 'structure' in template:
                    parts.append("    Structure:")
                    for struct in template['structure']:
                        parts.append(f"      - {struct}")
                if 'use_cases' in template:
                    parts.append(f"    Use Cases: {', '.join(template['use_cases'])}")
        
        if 'metrics_to_track' in item:
            parts.append("Metrics to Track:")
            for metric in item['metrics_to_track']:
                parts.append(f"  - {metric}")
        
        if 'platforms' in item:
            parts.append("Platforms:")
            for platform in item['platforms']:
                parts.append(f"  {platform.get('platform', '')}: {platform.get('best_for', '')}")
                if 'content_types' in platform:
                    parts.append(f"    Content Types: {', '.join(platform['content_types'])}")
        
        if 'campaign_phases' in item:
            parts.append("Campaign Phases:")
            for phase in item['campaign_phases']:
                parts.append(f"  {phase.get('phase', '')}: {phase.get('timeline', '')}")
                if 'activities' in phase:
                    for activity in phase['activities']:
                        parts.append(f"    - {activity}")
                
    elif item_type == "segmentation_targeting":
        if 'segmentation_criteria' in item:
            parts.append("Segmentation Criteria:")
            for criteria in item['segmentation_criteria']:
                parts.append(f"  {criteria.get('criteria', '')}: {criteria.get('description', '')}")
                if 'examples' in criteria:
                    parts.append(f"    Examples: {', '.join(criteria['examples'])}")
                if 'use_cases' in criteria:
                    parts.append(f"    Use Cases: {', '.join(criteria['use_cases'])}")
        
        if 'targeting_strategies' in item:
            parts.append("Targeting Strategies:")
            for strategy in item['targeting_strategies']:
                parts.append(f"  {strategy.get('strategy', '')}: {strategy.get('description', '')}")
                if 'method' in strategy:
                    parts.append(f"    Method: {strategy['method']}")
                if 'use_cases' in strategy:
                    parts.append(f"    Use Cases: {', '.join(strategy['use_cases'])}")
        
        if 'tools' in item:
            parts.append("Tools:")
            for tool in item['tools']:
                parts.append(f"  - {tool}")
        
        if 'best_practices' in item:
            parts.append("Best Practices:")
            for practice in item['best_practices']:
                parts.append(f"  - {practice}")
                
    elif item_type == "content_automation":
        if 'type' in item:
            parts.append(f"Content Type: {item.get('type', '')}")
        parts.append(f"Description: {item.get('description', '')}")
        
        if 'examples' in item:
            parts.append("Examples:")
            for example in item['examples']:
                parts.append(f"  - {example}")
        
        if 'automation_triggers' in item:
            parts.append("Automation Triggers:")
            for trigger in item['automation_triggers']:
                parts.append(f"  - {trigger}")
        
        if 'workflow_name' in item:
            parts.append(f"Workflow: {item.get('workflow_name', '')}")
            parts.append(f"Description: {item.get('description', '')}")
            if 'steps' in item:
                parts.append("Steps:")
                for step in item['steps']:
                    parts.append(f"  - {step}")
            if 'target_audience' in item:
                parts.append(f"Target Audience: {', '.join(item['target_audience'])}")
        
        if 'personalization_elements' in item:
            parts.append("Personalization Elements:")
            for element in item['personalization_elements']:
                parts.append(f"  - {element}")
        
        if 'dynamic_content' in item:
            parts.append("Dynamic Content:")
            for content in item['dynamic_content']:
                parts.append(f"  - {content}")
                
    elif item_type == "campaign_effectiveness_tracking":
        if 'metric_category' in item:
            parts.append(f"Metric Category: {item.get('metric_category', '')}")
        
        if 'metrics' in item:
            parts.append("Metrics:")
            for metric in item['metrics']:
                parts.append(f"  {metric.get('metric', '')}: {metric.get('description', '')}")
                if 'calculation' in metric:
                    parts.append(f"    Calculation: {metric['calculation']}")
                if 'benchmark' in metric:
                    parts.append(f"    Benchmark: {metric['benchmark']}")
                if 'improvement_tips' in metric:
                    parts.append("    Improvement Tips:")
                    for tip in metric['improvement_tips']:
                        parts.append(f"      - {tip}")
        
        if 'tools' in item:
            parts.append("Tools:")
            for tool in item['tools']:
                parts.append(f"  - {tool}")
        
        if 'tracking_setup' in item:
            parts.append("Tracking Setup:")
            for setup in item['tracking_setup']:
                parts.append(f"  - {setup}")
        
        if 'report_types' in item:
            parts.append("Report Types:")
            for report in item['report_types']:
                parts.append(f"  {report.get('report', '')}: {report.get('frequency', '')}")
                if 'includes' in report:
                    parts.append(f"    Includes: {', '.join(report['includes'])}")
                
    elif item_type == "doctor_journey_mapping":
        parts.append(f"Journey Stage: {item.get('stage', '')}")
        parts.append(f"Description: {item.get('description', '')}")
        
        if 'key_activities' in item:
            parts.append("Key Activities:")
            for activity in item['key_activities']:
                parts.append(f"  - {activity}")
        
        if 'engagement_tactics' in item:
            parts.append("Engagement Tactics:")
            for tactic in item['engagement_tactics']:
                parts.append(f"  - {tactic}")
        
        if 'success_metrics' in item:
            parts.append("Success Metrics:")
            for metric in item['success_metrics']:
                parts.append(f"  - {metric}")
        
        if 'optimization_strategies' in item:
            parts.append("Optimization Strategies:")
            for strategy in item['optimization_strategies']:
                parts.append(f"  - {strategy}")
        
        if 'tools' in item:
            parts.append("Tools:")
            for tool in item['tools']:
                parts.append(f"  - {tool}")
    
    text_blob = "\n".join(parts).strip()
    return text_blob[:6000]  # Keep within reasonable size

def create_embedding(text: str):
    """Generate an embedding vector for a given text."""
    if not text:
        text = " "  # avoid empty input to embeddings API
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def get_aws_region(region_string: str):
    """Get AwsRegion enum value from string, with fallback."""
    region_map = {
        "us-east-1": AwsRegion.US_EAST_1,
        "us-west-2": AwsRegion.US_WEST_2,
        "eu-west-1": AwsRegion.EU_WEST_1,
    }
    
    if region_string in region_map:
        return region_map[region_string]
    
    try:
        enum_name = region_string.replace("-", "_").upper()
        if hasattr(AwsRegion, enum_name):
            return getattr(AwsRegion, enum_name)
    except Exception:
        pass
    
    print(f"⚠️  Region '{region_string}' not found, defaulting to us-east-1")
    return AwsRegion.US_EAST_1

def store_embeddings():
    """Store embeddings in Pinecone index."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get embedding dimension from OpenAI model
    print(f"Getting embedding dimension for model: {EMBEDDING_MODEL}")
    sample_embedding = create_embedding("sample")
    embedding_dimension = len(sample_embedding)
    print(f"✅ Embedding dimension: {embedding_dimension}")
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        
        aws_region = get_aws_region(PINECONE_REGION)
        
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=aws_region
            )
        )
        print(f"✅ Index {PINECONE_INDEX_NAME} created successfully with dimension {embedding_dimension}!")
    else:
        print(f"✅ Index {PINECONE_INDEX_NAME} already exists")
        index_stats = pc.describe_index(PINECONE_INDEX_NAME)
        index_dimension = index_stats.dimension
        
        if index_dimension != embedding_dimension:
            raise ValueError(
                f"❌ Dimension mismatch!\n"
                f"   Index '{PINECONE_INDEX_NAME}' has dimension: {index_dimension}\n"
                f"   Embedding model '{EMBEDDING_MODEL}' produces dimension: {embedding_dimension}\n\n"
                f"   Solutions:\n"
                f"   1. Use a different index name (set PINECONE_INDEX_NAME in .env)\n"
                f"   2. Delete the existing index and recreate it\n"
                f"   3. Use an embedding model that matches the index dimension\n"
            )
        else:
            print(f"✅ Index dimension ({index_dimension}) matches embedding dimension ({embedding_dimension})")
    
    # Connect to index
    index = pc.Index(name=PINECONE_INDEX_NAME)
    
    # Load JSON
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Prepare vectors for batch upsert
    vectors_to_upsert = []
    
    doc_count = 0
    
    # Process Multi-channel Campaigns data
    if "multi_channel_campaigns" in data:
        print("\nProcessing Multi-channel Campaigns data...")
        for campaign_type, campaign_info in data["multi_channel_campaigns"].items():
            uid = f"campaign_{campaign_type}"
            text = build_text_for_embedding(campaign_info, "multi_channel_campaigns", campaign_type)
            embedding = create_embedding(text)
            
            metadata = {
                "type": "multi_channel_campaigns",
                "campaign_type": campaign_type,
                "channel": campaign_info.get("campaign_type", ""),
                "text": text
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            vectors_to_upsert.append({
                "id": uid,
                "values": embedding,
                "metadata": metadata
            })
            doc_count += 1
    
    # Process Segmentation & Targeting data
    if "segmentation_targeting" in data:
        print("\nProcessing Segmentation & Targeting data...")
        for segment_type, segment_info in data["segmentation_targeting"].items():
            uid = f"segmentation_{segment_type}"
            text = build_text_for_embedding(segment_info, "segmentation_targeting", segment_type)
            embedding = create_embedding(text)
            
            metadata = {
                "type": "segmentation_targeting",
                "segment_type": segment_type,
                "text": text
            }
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            vectors_to_upsert.append({
                "id": uid,
                "values": embedding,
                "metadata": metadata
            })
            doc_count += 1
    
    # Process Content Automation data
    if "content_automation" in data:
        print("\nProcessing Content Automation data...")
        content_automation = data["content_automation"]
        
        # Process content_types
        if "content_types" in content_automation:
            for content_type_name, content_type_info in content_automation["content_types"].items():
                uid = f"content_{content_type_name}"
                text = build_text_for_embedding(content_type_info, "content_automation", content_type_name)
                embedding = create_embedding(text)
                
                metadata = {
                    "type": "content_automation",
                    "category": "content_type",
                    "content_type": content_type_name,
                    "text": text
                }
                metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
                
                vectors_to_upsert.append({
                    "id": uid,
                    "values": embedding,
                    "metadata": metadata
                })
                doc_count += 1
        
        # Process automation_workflows
        if "automation_workflows" in content_automation:
            for workflow_name, workflow_info in content_automation["automation_workflows"].items():
                uid = f"workflow_{workflow_name}"
                text = build_text_for_embedding(workflow_info, "content_automation", workflow_name)
                embedding = create_embedding(text)
                
                metadata = {
                    "type": "content_automation",
                    "category": "workflow",
                    "workflow_name": workflow_name,
                    "text": text
                }
                metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
                
                vectors_to_upsert.append({
                    "id": uid,
                    "values": embedding,
                    "metadata": metadata
                })
                doc_count += 1
        
        # Process content_personalization
        if "content_personalization" in content_automation:
            personalization_info = content_automation["content_personalization"]
            uid = "personalization"
            text = build_text_for_embedding(personalization_info, "content_automation", "personalization")
            embedding = create_embedding(text)
            
            metadata = {
                "type": "content_automation",
                "category": "personalization",
                "text": text
            }
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            vectors_to_upsert.append({
                "id": uid,
                "values": embedding,
                "metadata": metadata
            })
            doc_count += 1
    
    # Process Campaign Effectiveness Tracking data
    if "campaign_effectiveness_tracking" in data:
        print("\nProcessing Campaign Effectiveness Tracking data...")
        for tracking_category, tracking_info in data["campaign_effectiveness_tracking"].items():
            if isinstance(tracking_info, dict):
                uid = f"tracking_{tracking_category}"
                text = build_text_for_embedding(tracking_info, "campaign_effectiveness_tracking", tracking_category)
                embedding = create_embedding(text)
                
                metadata = {
                    "type": "campaign_effectiveness_tracking",
                    "category": tracking_category,
                    "text": text
                }
                metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
                
                vectors_to_upsert.append({
                    "id": uid,
                    "values": embedding,
                    "metadata": metadata
                })
                doc_count += 1
    
    # Process Doctor Journey Mapping data
    if "doctor_journey_mapping" in data:
        print("\nProcessing Doctor Journey Mapping data...")
        if "journey_stages" in data["doctor_journey_mapping"]:
            for stage_name, stage_info in data["doctor_journey_mapping"]["journey_stages"].items():
                uid = f"journey_{stage_name}"
                text = build_text_for_embedding(stage_info, "doctor_journey_mapping", stage_name)
                embedding = create_embedding(text)
                
                metadata = {
                    "type": "doctor_journey_mapping",
                    "stage": stage_info.get("stage", stage_name),
                    "text": text
                }
                metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
                
                vectors_to_upsert.append({
                    "id": uid,
                    "values": embedding,
                    "metadata": metadata
                })
                doc_count += 1
        
        if "journey_optimization" in data["doctor_journey_mapping"]:
            opt_info = data["doctor_journey_mapping"]["journey_optimization"]
            uid = "journey_optimization"
            text = build_text_for_embedding(opt_info, "doctor_journey_mapping", "optimization")
            embedding = create_embedding(text)
            
            metadata = {
                "type": "doctor_journey_mapping",
                "category": "optimization",
                "text": text
            }
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            vectors_to_upsert.append({
                "id": uid,
                "values": embedding,
                "metadata": metadata
            })
            doc_count += 1
    
    print(f"\nTotal documents prepared: {doc_count}")
    
    # Batch upsert to Pinecone (upsert in batches of 100)
    batch_size = 100
    total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
    
    print(f"\nAdding documents to Pinecone in {total_batches} batches...")
    
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        batch_num = i // batch_size + 1
        print(f"   Upserted batch {batch_num}/{total_batches} ({len(batch)} documents)")
    
    print(f"\n✅ All embeddings stored successfully in Pinecone!")
    print(f"   Total documents: {doc_count}")

if __name__ == "__main__":
    store_embeddings()
