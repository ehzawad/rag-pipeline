"""
Real-world RAG evaluation system with industry-standard metrics and benchmarks.

This module provides comprehensive evaluation capabilities for RAG systems using
real-world datasets and scenarios commonly found in production environments.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    query: str
    expected_answer: str
    generated_answer: str
    retrieved_docs: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    retrieval_score: float
    relevance_score: float
    accuracy_score: float
    category: str

class RealWorldEvaluator:
    """
    Comprehensive evaluator for RAG systems using real-world scenarios.
    
    Supports multiple evaluation dimensions:
    - Retrieval accuracy and speed
    - Generation quality and relevance
    - End-to-end performance
    - Domain-specific metrics
    - Production readiness indicators
    """
    
    def __init__(self, rag_pipeline=None):
        """Initialize evaluator with RAG pipeline."""
        self.rag_pipeline = rag_pipeline
        self.results = []
        
        # Industry benchmark thresholds
        self.benchmarks = {
            "retrieval_time_ms": 100,  # Sub-100ms retrieval
            "generation_time_ms": 2000,  # Sub-2s generation
            "total_time_ms": 3000,  # Sub-3s total response
            "retrieval_accuracy": 0.85,  # 85%+ retrieval accuracy
            "answer_relevance": 0.80,  # 80%+ answer relevance
            "answer_accuracy": 0.75,  # 75%+ factual accuracy
        }
    
    def load_evaluation_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load evaluation questions from JSON file."""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            logger.info(f"Loaded {len(questions)} evaluation questions from {dataset_path}")
            return questions
        except Exception as e:
            logger.error(f"Error loading evaluation dataset: {e}")
            return []
    
    def evaluate_retrieval_accuracy(self, retrieved_docs: List[Dict], expected_docs: List[str]) -> float:
        """
        Calculate retrieval accuracy using precision@k.
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            expected_docs: List of expected relevant document names
            
        Returns:
            Retrieval accuracy score (0-1)
        """
        if not retrieved_docs or not expected_docs:
            return 0.0
        
        # Extract document names from retrieved docs
        retrieved_names = []
        for doc in retrieved_docs:
            if 'metadata' in doc and 'source' in doc['metadata']:
                source = doc['metadata']['source']
                # Extract filename from path
                filename = Path(source).name
                retrieved_names.append(filename)
        
        # Calculate precision@k
        relevant_retrieved = len(set(retrieved_names) & set(expected_docs))
        precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0
        
        return precision
    
    def evaluate_answer_relevance(self, question: str, answer: str, context: str) -> float:
        """
        Evaluate answer relevance using simple heuristics.
        
        In production, this would use more sophisticated NLP models.
        """
        if not answer or answer.strip() == "":
            return 0.0
        
        # Simple keyword overlap scoring
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        question_words -= stop_words
        answer_words -= stop_words
        
        # Calculate relevance score
        if not question_words:
            return 0.5  # Neutral score if no meaningful words
        
        overlap = len(question_words & answer_words)
        relevance = overlap / len(question_words)
        
        # Bonus for using context information
        context_overlap = len(answer_words & context_words) / len(context_words) if context_words else 0
        relevance += 0.2 * context_overlap
        
        return min(relevance, 1.0)
    
    def evaluate_answer_accuracy(self, generated_answer: str, expected_answer: str) -> float:
        """
        Evaluate factual accuracy of generated answer.
        
        Uses simple string matching and keyword extraction.
        In production, would use semantic similarity models.
        """
        if not generated_answer or not expected_answer:
            return 0.0
        
        generated_lower = generated_answer.lower()
        expected_lower = expected_answer.lower()
        
        # Extract key facts (numbers, percentages, company names, etc.)
        import re
        
        # Extract numbers and percentages
        generated_numbers = set(re.findall(r'\d+\.?\d*%?', generated_answer))
        expected_numbers = set(re.findall(r'\d+\.?\d*%?', expected_answer))
        
        # Extract key terms (capitalized words, likely proper nouns)
        generated_terms = set(re.findall(r'\b[A-Z][a-z]+\b', generated_answer))
        expected_terms = set(re.findall(r'\b[A-Z][a-z]+\b', expected_answer))
        
        # Calculate accuracy based on fact overlap
        number_accuracy = len(generated_numbers & expected_numbers) / len(expected_numbers) if expected_numbers else 1.0
        term_accuracy = len(generated_terms & expected_terms) / len(expected_terms) if expected_terms else 1.0
        
        # Simple substring matching for key phrases
        substring_score = 0.0
        if expected_lower in generated_lower or generated_lower in expected_lower:
            substring_score = 0.5
        
        # Combined accuracy score
        accuracy = (number_accuracy + term_accuracy + substring_score) / 3
        return min(accuracy, 1.0)
    
    def run_single_evaluation(self, question_data: Dict[str, Any]) -> EvaluationResult:
        """Run evaluation for a single question."""
        question = question_data['question']
        expected_answer = question_data['expected_answer']
        expected_docs = question_data.get('relevant_docs', [])
        category = question_data.get('category', 'general')
        
        logger.info(f"Evaluating: {question[:50]}...")
        
        # Measure retrieval time
        start_time = time.time()
        
        try:
            # Use contextual retrieval
            retrieved_docs = self.rag_pipeline.contextual_retriever.retrieve_with_context(
                question, k=5
            )
            retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Measure generation time
            gen_start = time.time()
            
            # Generate response
            context = "\n".join([doc.get('content', '') if isinstance(doc, dict) else str(doc) for doc in retrieved_docs])
            response_result = self.rag_pipeline.generator.generate_response(
                question, context, temperature=0.1, max_tokens=200
            )
            response = response_result.get('response', '') if isinstance(response_result, dict) else str(response_result)
            generation_time = (time.time() - gen_start) * 1000
            
            total_time = retrieval_time + generation_time
            
            # Calculate metrics
            retrieval_score = self.evaluate_retrieval_accuracy(retrieved_docs, expected_docs)
            relevance_score = self.evaluate_answer_relevance(question, response, context)
            accuracy_score = self.evaluate_answer_accuracy(response, expected_answer)
            
            return EvaluationResult(
                query=question,
                expected_answer=expected_answer,
                generated_answer=response,
                retrieved_docs=retrieved_docs,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                retrieval_score=retrieval_score,
                relevance_score=relevance_score,
                accuracy_score=accuracy_score,
                category=category
            )
            
        except Exception as e:
            logger.error(f"Error evaluating question: {e}")
            return EvaluationResult(
                query=question,
                expected_answer=expected_answer,
                generated_answer="ERROR: " + str(e),
                retrieved_docs=[],
                retrieval_time=0.0,
                generation_time=0.0,
                total_time=0.0,
                retrieval_score=0.0,
                relevance_score=0.0,
                accuracy_score=0.0,
                category=category
            )
    
    def run_comprehensive_evaluation(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Run comprehensive evaluation across multiple datasets."""
        logger.info("Starting comprehensive RAG evaluation...")
        
        all_results = []
        
        for dataset_path in dataset_paths:
            questions = self.load_evaluation_dataset(dataset_path)
            
            for question_data in questions:
                result = self.run_single_evaluation(question_data)
                all_results.append(result)
                self.results.append(result)
        
        # Calculate aggregate metrics
        metrics = self.calculate_aggregate_metrics(all_results)
        
        # Compare against benchmarks
        benchmark_comparison = self.compare_against_benchmarks(metrics)
        
        # Generate detailed report
        report = {
            "total_questions": len(all_results),
            "aggregate_metrics": metrics,
            "benchmark_comparison": benchmark_comparison,
            "category_breakdown": self.analyze_by_category(all_results),
            "performance_distribution": self.analyze_performance_distribution(all_results),
            "recommendations": self.generate_recommendations(metrics, benchmark_comparison)
        }
        
        logger.info("Comprehensive evaluation completed")
        return report
    
    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}
        
        metrics = {
            "avg_retrieval_time_ms": np.mean([r.retrieval_time for r in results]),
            "avg_generation_time_ms": np.mean([r.generation_time for r in results]),
            "avg_total_time_ms": np.mean([r.total_time for r in results]),
            "avg_retrieval_accuracy": np.mean([r.retrieval_score for r in results]),
            "avg_answer_relevance": np.mean([r.relevance_score for r in results]),
            "avg_answer_accuracy": np.mean([r.accuracy_score for r in results]),
            "p95_total_time_ms": np.percentile([r.total_time for r in results], 95),
            "p99_total_time_ms": np.percentile([r.total_time for r in results], 99),
            "success_rate": len([r for r in results if not r.generated_answer.startswith("ERROR")]) / len(results)
        }
        
        return metrics
    
    def compare_against_benchmarks(self, metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Compare metrics against industry benchmarks."""
        comparison = {}
        
        benchmark_mapping = {
            "avg_retrieval_time_ms": "retrieval_time_ms",
            "avg_generation_time_ms": "generation_time_ms", 
            "avg_total_time_ms": "total_time_ms",
            "avg_retrieval_accuracy": "retrieval_accuracy",
            "avg_answer_relevance": "answer_relevance",
            "avg_answer_accuracy": "answer_accuracy"
        }
        
        for metric_key, benchmark_key in benchmark_mapping.items():
            if metric_key in metrics and benchmark_key in self.benchmarks:
                actual = metrics[metric_key]
                target = self.benchmarks[benchmark_key]
                
                # For time metrics, lower is better
                if "time" in metric_key:
                    meets_benchmark = actual <= target
                    performance_ratio = target / actual if actual > 0 else float('inf')
                else:
                    meets_benchmark = actual >= target
                    performance_ratio = actual / target if target > 0 else 0
                
                comparison[metric_key] = {
                    "actual": actual,
                    "target": target,
                    "meets_benchmark": meets_benchmark,
                    "performance_ratio": performance_ratio,
                    "status": "PASS" if meets_benchmark else "FAIL"
                }
        
        return comparison
    
    def analyze_by_category(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by question category."""
        categories = {}
        
        for result in results:
            category = result.category
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        category_metrics = {}
        for category, cat_results in categories.items():
            category_metrics[category] = self.calculate_aggregate_metrics(cat_results)
        
        return category_metrics
    
    def analyze_performance_distribution(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze performance distribution and identify outliers."""
        times = [r.total_time for r in results]
        accuracies = [r.accuracy_score for r in results]
        
        return {
            "time_distribution": {
                "min": min(times),
                "max": max(times),
                "median": np.median(times),
                "std": np.std(times)
            },
            "accuracy_distribution": {
                "min": min(accuracies),
                "max": max(accuracies),
                "median": np.median(accuracies),
                "std": np.std(accuracies)
            },
            "slow_queries": len([t for t in times if t > self.benchmarks["total_time_ms"]]),
            "low_accuracy_queries": len([a for a in accuracies if a < self.benchmarks["answer_accuracy"]])
        }
    
    def generate_recommendations(self, metrics: Dict[str, float], benchmarks: Dict[str, Dict]) -> List[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []
        
        # Performance recommendations
        if benchmarks.get("avg_total_time_ms", {}).get("status") == "FAIL":
            recommendations.append("PERFORMANCE: Consider optimizing retrieval speed or implementing caching")
        
        if benchmarks.get("avg_retrieval_accuracy", {}).get("status") == "FAIL":
            recommendations.append("RETRIEVAL: Improve document chunking strategy or embedding model")
        
        if benchmarks.get("avg_answer_accuracy", {}).get("status") == "FAIL":
            recommendations.append("GENERATION: Fine-tune prompts or consider using a larger language model")
        
        # Success rate recommendations
        if metrics.get("success_rate", 1.0) < 0.95:
            recommendations.append("RELIABILITY: Implement better error handling and fallback mechanisms")
        
        # General recommendations
        if metrics.get("avg_answer_relevance", 0) < 0.8:
            recommendations.append("RELEVANCE: Enhance context selection and prompt engineering")
        
        if not recommendations:
            recommendations.append("EXCELLENT: System meets all industry benchmarks!")
        
        return recommendations
    
    def save_detailed_report(self, report: Dict[str, Any], output_path: str):
        """Save detailed evaluation report to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        report_serializable = convert_numpy(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2)
        
        logger.info(f"Detailed report saved to {output_path}")
    
    def create_performance_visualizations(self, output_dir: str):
        """Create performance visualization charts."""
        if not self.results:
            logger.warning("No results available for visualization")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Response time distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        times = [r.total_time for r in self.results]
        plt.hist(times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.benchmarks["total_time_ms"], color='red', linestyle='--', label='Benchmark')
        plt.xlabel('Total Response Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')
        plt.legend()
        
        # 2. Accuracy scores by category
        plt.subplot(2, 2, 2)
        categories = [r.category for r in self.results]
        accuracies = [r.accuracy_score for r in self.results]
        
        category_data = {}
        for cat, acc in zip(categories, accuracies):
            if cat not in category_data:
                category_data[cat] = []
            category_data[cat].append(acc)
        
        plt.boxplot(category_data.values(), labels=category_data.keys())
        plt.axhline(self.benchmarks["answer_accuracy"], color='red', linestyle='--', label='Benchmark')
        plt.ylabel('Accuracy Score')
        plt.title('Accuracy by Category')
        plt.xticks(rotation=45)
        plt.legend()
        
        # 3. Retrieval vs Generation time
        plt.subplot(2, 2, 3)
        retrieval_times = [r.retrieval_time for r in self.results]
        generation_times = [r.generation_time for r in self.results]
        
        plt.scatter(retrieval_times, generation_times, alpha=0.6, color='green')
        plt.xlabel('Retrieval Time (ms)')
        plt.ylabel('Generation Time (ms)')
        plt.title('Retrieval vs Generation Time')
        
        # 4. Performance correlation
        plt.subplot(2, 2, 4)
        relevance_scores = [r.relevance_score for r in self.results]
        accuracy_scores = [r.accuracy_score for r in self.results]
        
        plt.scatter(relevance_scores, accuracy_scores, alpha=0.6, color='purple')
        plt.xlabel('Relevance Score')
        plt.ylabel('Accuracy Score')
        plt.title('Relevance vs Accuracy Correlation')
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance visualizations saved to {output_path}")

def main():
    """Example usage of the real-world evaluator."""
    # This would be called with an actual RAG pipeline
    print("Real-world RAG Evaluator")
    print("========================")
    print("This module provides comprehensive evaluation capabilities for RAG systems.")
    print("Usage: Import and use with your RAG pipeline for production-ready evaluation.")

if __name__ == "__main__":
    main()
