"""
Run comprehensive real-world evaluation of the RAG pipeline.

This script demonstrates the RAG system's performance on industry-standard
datasets and scenarios, providing detailed metrics and benchmarks.
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from contextual_retrieval import ContextualRetrieval
from real_world_evaluation import RealWorldEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run comprehensive real-world evaluation."""
    logger.info("🚀 Starting Real-World RAG Pipeline Evaluation")
    logger.info("=" * 60)
    
    try:
        # Initialize the RAG pipeline with contextual retrieval
        logger.info("Initializing RAG pipeline with contextual retrieval...")
        
        # Build knowledge base with real-world data
        contextual_retriever = ContextualRetrieval()
        
        # Check if knowledge base exists, if not build it
        data_dir = "./data/real_world_datasets"
        if not Path("./data/vector_store/chroma.sqlite3").exists():
            logger.info("Building knowledge base from real-world datasets...")
            contextual_retriever.build_contextual_knowledge_base(data_dir)
        else:
            logger.info("Using existing knowledge base...")
        
        # Initialize evaluator
        evaluator = RealWorldEvaluator(rag_pipeline=type('Pipeline', (), {
            'contextual_retriever': contextual_retriever,
            'generator': contextual_retriever.contextualizer
        })())
        
        # Run comprehensive evaluation
        logger.info("🔍 Running comprehensive evaluation...")
        
        dataset_paths = [
            "./data/real_world_datasets/financial_eval_questions.json",
            "./data/real_world_datasets/support_eval_questions.json"
        ]
        
        # Execute evaluation
        report = evaluator.run_comprehensive_evaluation(dataset_paths)
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("📊 EVALUATION RESULTS")
        logger.info("=" * 60)
        
        # Overall metrics
        metrics = report["aggregate_metrics"]
        logger.info(f"📈 Total Questions Evaluated: {report['total_questions']}")
        logger.info(f"⚡ Average Response Time: {metrics['avg_total_time_ms']:.1f}ms")
        logger.info(f"🎯 Average Retrieval Accuracy: {metrics['avg_retrieval_accuracy']:.2%}")
        logger.info(f"💡 Average Answer Relevance: {metrics['avg_answer_relevance']:.2%}")
        logger.info(f"✅ Average Answer Accuracy: {metrics['avg_answer_accuracy']:.2%}")
        logger.info(f"🔄 Success Rate: {metrics['success_rate']:.2%}")
        
        # Benchmark comparison
        logger.info("\n🏆 BENCHMARK COMPARISON")
        logger.info("-" * 30)
        
        benchmarks = report["benchmark_comparison"]
        for metric, data in benchmarks.items():
            status_emoji = "✅" if data["status"] == "PASS" else "❌"
            logger.info(f"{status_emoji} {metric}: {data['actual']:.1f} (target: {data['target']:.1f}) - {data['status']}")
        
        # Category breakdown
        logger.info("\n📋 PERFORMANCE BY CATEGORY")
        logger.info("-" * 30)
        
        for category, cat_metrics in report["category_breakdown"].items():
            logger.info(f"🏷️  {category.upper()}:")
            logger.info(f"   • Accuracy: {cat_metrics['avg_answer_accuracy']:.2%}")
            logger.info(f"   • Relevance: {cat_metrics['avg_answer_relevance']:.2%}")
            logger.info(f"   • Response Time: {cat_metrics['avg_total_time_ms']:.1f}ms")
        
        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS")
        logger.info("-" * 20)
        
        for i, rec in enumerate(report["recommendations"], 1):
            logger.info(f"{i}. {rec}")
        
        # Save detailed report
        report_path = "./data/evaluation_results_detailed.json"
        evaluator.save_detailed_report(report, report_path)
        logger.info(f"\n📄 Detailed report saved to: {report_path}")
        
        # Create visualizations
        logger.info("📊 Generating performance visualizations...")
        evaluator.create_performance_visualizations("./data/evaluation_charts")
        logger.info("📈 Charts saved to: ./data/evaluation_charts/")
        
        # Production readiness assessment
        logger.info("\n🏭 PRODUCTION READINESS ASSESSMENT")
        logger.info("-" * 40)
        
        production_score = 0
        total_checks = 6
        
        # Check each benchmark
        for metric, data in benchmarks.items():
            if data["status"] == "PASS":
                production_score += 1
        
        readiness_percentage = (production_score / total_checks) * 100
        
        if readiness_percentage >= 85:
            readiness_status = "🟢 PRODUCTION READY"
        elif readiness_percentage >= 70:
            readiness_status = "🟡 NEEDS OPTIMIZATION"
        else:
            readiness_status = "🔴 NOT PRODUCTION READY"
        
        logger.info(f"Production Readiness Score: {readiness_percentage:.1f}% ({production_score}/{total_checks})")
        logger.info(f"Status: {readiness_status}")
        
        # Industry comparison
        logger.info("\n🌐 INDUSTRY COMPARISON")
        logger.info("-" * 25)
        logger.info("Compared to industry standards:")
        logger.info("• Sub-100ms retrieval: Industry standard for real-time systems")
        logger.info("• Sub-3s total response: User experience benchmark")
        logger.info("• 85%+ retrieval accuracy: Enterprise RAG systems")
        logger.info("• 75%+ answer accuracy: Production quality threshold")
        
        # Sample results
        logger.info("\n🔍 SAMPLE EVALUATION RESULTS")
        logger.info("-" * 35)
        
        for i, result in enumerate(evaluator.results[:3], 1):
            logger.info(f"\nExample {i}:")
            logger.info(f"❓ Question: {result.query}")
            logger.info(f"✅ Expected: {result.expected_answer}")
            logger.info(f"🤖 Generated: {result.generated_answer[:100]}...")
            logger.info(f"⏱️  Time: {result.total_time:.1f}ms")
            logger.info(f"📊 Scores: Accuracy={result.accuracy_score:.2f}, Relevance={result.relevance_score:.2f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Real-world evaluation completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
