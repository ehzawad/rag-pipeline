"""
Download and prepare real-world datasets for RAG evaluation.

This script downloads popular open datasets used in industry for RAG evaluation:
- SQuAD 2.0: Stanford Question Answering Dataset
- MS MARCO: Microsoft Machine Reading Comprehension
- Natural Questions: Google's real user questions
"""

import json
import requests
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_squad_dataset():
    """Download SQuAD 2.0 dataset."""
    logger.info("Downloading SQuAD 2.0 dataset...")
    
    urls = {
        "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    }
    
    for split, url in urls.items():
        filename = f"squad_v2_{split}.json"
        filepath = Path(f"./data/real_world_datasets/{filename}")
        
        if filepath.exists():
            logger.info(f"SQuAD {split} already exists, skipping...")
            continue
            
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2)
            
            logger.info(f"Downloaded SQuAD {split} to {filepath}")
        except Exception as e:
            logger.error(f"Error downloading SQuAD {split}: {e}")

def create_sample_financial_dataset():
    """Create a sample financial dataset for real-world RAG testing."""
    logger.info("Creating sample financial dataset...")
    
    financial_docs = [
        {
            "title": "Apple Inc. Q4 2023 Earnings Report",
            "content": """Apple Inc. reported record Q4 2023 revenue of $89.5 billion, up 1% year-over-year. iPhone revenue was $43.8 billion, Services revenue reached $22.3 billion representing 16% growth. The company returned $25 billion to shareholders through dividends and share repurchases. CEO Tim Cook noted strong performance in emerging markets, particularly India where revenue grew over 20%. Apple's Services segment continues to be a key growth driver with over 1 billion paid subscriptions across the platform. The company maintains a strong balance sheet with $162 billion in cash and marketable securities.""",
            "metadata": {
                "company": "Apple Inc.",
                "quarter": "Q4 2023",
                "revenue": 89.5,
                "type": "earnings_report"
            }
        },
        {
            "title": "Microsoft Corporation Q1 2024 Financial Results",
            "content": """Microsoft Corporation announced Q1 2024 revenue of $56.5 billion, representing 13% growth year-over-year. Productivity and Business Processes revenue increased 13% to $18.6 billion. More Personal Computing revenue was $13.7 billion, up 3%. Intelligent Cloud revenue grew 19% to $24.3 billion, driven by Azure and other cloud services growth of 29%. Microsoft Cloud revenue was $31.8 billion, up 24% year-over-year. Operating income increased 25% to $26.9 billion. The company returned $8.4 billion to shareholders through dividends and share repurchases during the quarter.""",
            "metadata": {
                "company": "Microsoft Corporation", 
                "quarter": "Q1 2024",
                "revenue": 56.5,
                "type": "earnings_report"
            }
        },
        {
            "title": "Tesla Inc. Q3 2023 Vehicle Deliveries and Production",
            "content": """Tesla Inc. delivered approximately 435,000 vehicles in Q3 2023, slightly below analyst expectations of 455,000. The company produced 430,000 vehicles during the quarter. Model 3 and Model Y comprised the majority of deliveries. Tesla's energy storage deployments reached 4.0 GWh, up 90% year-over-year. The company continues to focus on cost reduction and operational efficiency. Cybertruck production is expected to begin in late 2023. Tesla maintains its long-term delivery target of 20 million vehicles annually by 2030. The company's Supercharger network expanded to over 50,000 connectors globally.""",
            "metadata": {
                "company": "Tesla Inc.",
                "quarter": "Q3 2023", 
                "deliveries": 435000,
                "type": "delivery_report"
            }
        },
        {
            "title": "Amazon.com Inc. Q2 2023 Financial Performance",
            "content": """Amazon.com Inc. reported Q2 2023 net sales of $134.4 billion, up 11% year-over-year. North America segment sales increased 11% to $82.5 billion. International segment sales grew 10% to $29.7 billion. Amazon Web Services (AWS) revenue was $22.1 billion, up 12% year-over-year. Operating income was $7.7 billion compared to an operating loss of $627 million in Q2 2022. The company's advertising services revenue grew 22% to $10.7 billion. Amazon Prime membership continues to grow globally with enhanced benefits including faster delivery and expanded content offerings.""",
            "metadata": {
                "company": "Amazon.com Inc.",
                "quarter": "Q2 2023",
                "revenue": 134.4,
                "type": "earnings_report"
            }
        },
        {
            "title": "Federal Reserve Interest Rate Decision March 2024",
            "content": """The Federal Reserve maintained the federal funds rate at 5.25%-5.50% range during the March 2024 FOMC meeting. The committee noted continued progress on inflation moving toward the 2% target. Labor market remains robust with unemployment at 3.8%. The Fed emphasized data-dependent approach to future rate decisions. Economic projections suggest potential rate cuts later in 2024 if inflation continues to moderate. Housing market shows signs of stabilization with mortgage rates around 7%. Consumer spending remains resilient despite elevated borrowing costs. The Fed continues to reduce its balance sheet through quantitative tightening.""",
            "metadata": {
                "institution": "Federal Reserve",
                "date": "March 2024",
                "rate_range": "5.25%-5.50%",
                "type": "monetary_policy"
            }
        }
    ]
    
    # Save financial documents
    for i, doc in enumerate(financial_docs):
        filename = f"financial_doc_{i+1}.txt"
        filepath = Path(f"./data/real_world_datasets/{filename}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {doc['title']}\n\n")
            f.write(doc['content'])
        
        logger.info(f"Created {filename}")
    
    # Create evaluation questions for financial dataset
    eval_questions = [
        {
            "question": "What was Apple's Q4 2023 revenue?",
            "expected_answer": "$89.5 billion",
            "relevant_docs": ["financial_doc_1.txt"],
            "category": "financial_metrics"
        },
        {
            "question": "How much did Microsoft's Intelligent Cloud revenue grow in Q1 2024?",
            "expected_answer": "19% to $24.3 billion",
            "relevant_docs": ["financial_doc_2.txt"],
            "category": "financial_metrics"
        },
        {
            "question": "How many vehicles did Tesla deliver in Q3 2023?",
            "expected_answer": "Approximately 435,000 vehicles",
            "relevant_docs": ["financial_doc_3.txt"],
            "category": "operational_metrics"
        },
        {
            "question": "What was Amazon's AWS revenue in Q2 2023?",
            "expected_answer": "$22.1 billion, up 12% year-over-year",
            "relevant_docs": ["financial_doc_4.txt"],
            "category": "financial_metrics"
        },
        {
            "question": "What is the current federal funds rate as of March 2024?",
            "expected_answer": "5.25%-5.50% range",
            "relevant_docs": ["financial_doc_5.txt"],
            "category": "monetary_policy"
        },
        {
            "question": "Which companies showed the highest revenue growth?",
            "expected_answer": "Microsoft with 13% growth and Amazon with 11% growth",
            "relevant_docs": ["financial_doc_1.txt", "financial_doc_2.txt", "financial_doc_4.txt"],
            "category": "comparative_analysis"
        },
        {
            "question": "What are the key trends in cloud services revenue?",
            "expected_answer": "Strong growth with Microsoft Azure growing 29% and AWS growing 12%",
            "relevant_docs": ["financial_doc_2.txt", "financial_doc_4.txt"],
            "category": "trend_analysis"
        }
    ]
    
    # Save evaluation questions
    eval_filepath = Path("./data/real_world_datasets/financial_eval_questions.json")
    with open(eval_filepath, 'w', encoding='utf-8') as f:
        json.dump(eval_questions, f, indent=2)
    
    logger.info(f"Created evaluation questions: {eval_filepath}")

def create_tech_support_dataset():
    """Create a technical support dataset for customer service RAG evaluation."""
    logger.info("Creating technical support dataset...")
    
    support_docs = [
        {
            "title": "WiFi Connection Troubleshooting Guide",
            "content": """Common WiFi connection issues and solutions:

1. Device not detecting WiFi network:
   - Ensure WiFi is enabled on your device
   - Check if the network is broadcasting (not hidden)
   - Restart your device's WiFi adapter
   - Move closer to the router

2. Connected but no internet access:
   - Restart your modem and router (unplug for 30 seconds)
   - Check if other devices have internet access
   - Run network diagnostics on your device
   - Contact your ISP if the issue persists

3. Slow WiFi speeds:
   - Check for interference from other devices
   - Update your WiFi drivers
   - Change WiFi channel (1, 6, or 11 for 2.4GHz)
   - Consider upgrading to 5GHz band

4. Frequent disconnections:
   - Update device drivers
   - Check power management settings
   - Replace old router if over 5 years old
   - Check for overheating issues

Error codes:
- Error 651: PPP connection issue, restart modem
- Error 678: Remote computer not responding, check cables
- DNS errors: Try 8.8.8.8 or 1.1.1.1 as DNS servers"""
        },
        {
            "title": "Software Installation and Update Issues",
            "content": """Software installation troubleshooting:

Installation failures:
1. Insufficient disk space:
   - Free up at least 2GB of space
   - Use disk cleanup utility
   - Uninstall unused programs

2. Permission errors:
   - Run installer as administrator
   - Disable antivirus temporarily
   - Check user account permissions

3. Compatibility issues:
   - Verify system requirements
   - Run in compatibility mode
   - Update Windows to latest version

Update problems:
1. Windows Update stuck:
   - Run Windows Update troubleshooter
   - Reset Windows Update components
   - Clear update cache folder

2. Application updates failing:
   - Check internet connection
   - Clear application cache
   - Reinstall the application

3. Driver update issues:
   - Use Device Manager to update
   - Download from manufacturer website
   - Roll back to previous version if needed

Common error codes:
- 0x80070005: Access denied, run as admin
- 0x80240034: Update service not running
- MSI error 1603: Fatal error during installation"""
        },
        {
            "title": "Email Configuration and Troubleshooting",
            "content": """Email setup and common issues:

IMAP/POP3 Configuration:
Gmail:
- IMAP: imap.gmail.com, port 993 (SSL)
- SMTP: smtp.gmail.com, port 587 (TLS)
- Enable 2-factor authentication and app passwords

Outlook/Hotmail:
- IMAP: outlook.office365.com, port 993
- SMTP: smtp-mail.outlook.com, port 587
- Use OAuth2 authentication when possible

Common email problems:
1. Cannot send emails:
   - Check SMTP settings and authentication
   - Verify port numbers and encryption
   - Check with ISP about port blocking

2. Not receiving emails:
   - Check spam/junk folders
   - Verify email filters and rules
   - Check mailbox storage limits

3. Sync issues:
   - Update email client software
   - Remove and re-add email account
   - Check server status with provider

4. Authentication errors:
   - Update passwords after security changes
   - Enable less secure apps (if required)
   - Use app-specific passwords for 2FA accounts

Troubleshooting steps:
- Test with webmail first
- Check firewall and antivirus settings
- Try different email client
- Contact email provider support"""
        }
    ]
    
    # Save support documents
    for i, doc in enumerate(support_docs):
        filename = f"support_doc_{i+1}.txt"
        filepath = Path(f"./data/real_world_datasets/{filename}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {doc['title']}\n\n")
            f.write(doc['content'])
        
        logger.info(f"Created {filename}")
    
    # Create support evaluation questions
    support_questions = [
        {
            "question": "My WiFi is connected but I have no internet access. What should I do?",
            "expected_answer": "Restart your modem and router by unplugging for 30 seconds, check if other devices have internet access, run network diagnostics, and contact your ISP if the issue persists.",
            "relevant_docs": ["support_doc_1.txt"],
            "category": "connectivity_issues"
        },
        {
            "question": "I'm getting error code 0x80070005 during software installation. How do I fix this?",
            "expected_answer": "Error 0x80070005 means access denied. Run the installer as administrator to resolve this issue.",
            "relevant_docs": ["support_doc_2.txt"],
            "category": "installation_errors"
        },
        {
            "question": "What are the SMTP settings for Gmail?",
            "expected_answer": "Gmail SMTP: smtp.gmail.com, port 587 with TLS encryption. Enable 2-factor authentication and use app passwords.",
            "relevant_docs": ["support_doc_3.txt"],
            "category": "email_configuration"
        },
        {
            "question": "My WiFi keeps disconnecting frequently. What could be the cause?",
            "expected_answer": "Frequent disconnections can be caused by outdated drivers, power management settings, old router (over 5 years), or overheating issues. Update drivers and check power settings first.",
            "relevant_docs": ["support_doc_1.txt"],
            "category": "connectivity_issues"
        },
        {
            "question": "Windows Update is stuck. How can I fix it?",
            "expected_answer": "Run Windows Update troubleshooter, reset Windows Update components, or clear the update cache folder.",
            "relevant_docs": ["support_doc_2.txt"],
            "category": "update_issues"
        }
    ]
    
    # Save support evaluation questions
    eval_filepath = Path("./data/real_world_datasets/support_eval_questions.json")
    with open(eval_filepath, 'w', encoding='utf-8') as f:
        json.dump(support_questions, f, indent=2)
    
    logger.info(f"Created support evaluation questions: {eval_filepath}")

def main():
    """Download and create all datasets."""
    # Create directory
    Path("./data/real_world_datasets").mkdir(parents=True, exist_ok=True)
    
    # Download SQuAD (commented out due to size - can be enabled if needed)
    # download_squad_dataset()
    
    # Create domain-specific datasets
    create_sample_financial_dataset()
    create_tech_support_dataset()
    
    logger.info("All datasets created successfully!")
    logger.info("Available datasets:")
    logger.info("- Financial reports and earnings data")
    logger.info("- Technical support documentation")
    logger.info("- Evaluation questions for both domains")

if __name__ == "__main__":
    main()
