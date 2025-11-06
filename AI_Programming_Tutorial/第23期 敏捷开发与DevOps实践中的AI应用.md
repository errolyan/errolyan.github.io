# 第23期 敏捷开发与DevOps实践中的AI应用

欢迎回到AI编程深度专研系列教程！在上一期中，我们深入探讨了安全编码实践与AI辅助安全审计，学习了如何利用AI进行代码安全漏洞检测、安全审计以及安全事件响应。本期我们将继续第六章的内容，聚焦于如何在敏捷开发与DevOps实践中应用AI技术，提高开发效率和交付质量。

## 6.6.1 AI辅助的需求分析与用户故事生成

在敏捷开发中，需求分析和用户故事生成是至关重要的初始步骤。AI可以帮助团队更高效地完成这些任务，确保需求的质量和完整性。

### 智能需求分析工具

以下是一个使用AI进行智能需求分析的脚本示例：

```python
#!/usr/bin/env python3
# ai_requirement_analyzer.py - 使用AI进行智能需求分析

import os
import sys
import openai
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 配置OpenAI API密钥
openai.api_key = os.environ.get("OPENAI_API_KEY")

class AIRequirementAnalyzer:
    """
    智能需求分析器，使用AI分析需求文档并提供改进建议
    """
    
    @staticmethod
    def analyze_requirement_document(document_path: str) -> Dict[str, Any]:
        """
        分析需求文档并提供结构化分析
        """
        try:
            # 读取文档内容
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 限制内容大小
            max_content_length = 5000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "\n[内容过长，已截断]"
            
            # 构建提示词
            prompt = f"""
请作为一位产品管理和需求分析专家，分析以下需求文档。

文档路径: {document_path}

文档内容:
```
{content}
```

请提供以下分析：
1. 需求清晰度评估
2. 需求完整性检查（识别可能缺失的重要信息）
3. 需求歧义分析（找出可能有多种解释的部分）
4. 需求可行性和优先级建议
5. 将需求拆分为具体用户故事的建议
6. 潜在的风险和挑战
7. 改进建议

请以JSON格式返回分析结果，包含以下字段：
- clarity_score: 清晰度评分（0-100）
- completeness_score: 完整性评分（0-100）
- identified_ambiguities: 发现的歧义列表
- missing_requirements: 缺失的需求列表
- suggested_user_stories: 建议的用户故事列表，每个故事包含：
  - title: 标题
  - description: 描述
  - acceptance_criteria: 验收标准列表
  - priority: 优先级（High, Medium, Low）
- risks_and_challenges: 识别的风险和挑战列表
- improvement_suggestions: 改进建议列表
- executive_summary: 总体分析摘要

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API进行需求分析
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位资深的产品管理和需求分析专家，擅长分析各种类型的需求文档并提供专业的建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 添加文档信息
            result['document_path'] = document_path
            
            return result
            
        except Exception as e:
            print(f"分析需求文档失败 {document_path}: {e}")
            return {
                'document_path': document_path,
                'clarity_score': 0,
                'completeness_score': 0,
                'identified_ambiguities': [],
                'missing_requirements': [],
                'suggested_user_stories': [],
                'risks_and_challenges': [],
                'improvement_suggestions': [],
                'executive_summary': f"分析失败: {str(e)}"
            }
    
    @staticmethod
    def generate_user_stories(requirement_text: str, project_context: str = "") -> List[Dict[str, Any]]:
        """
        根据需求文本生成结构化的用户故事
        """
        try:
            # 构建提示词
            prompt = f"""
请根据以下需求文本和项目上下文，生成结构化的用户故事。

项目上下文: {project_context}

需求文本:
```
{requirement_text}
```

请将需求拆分为多个独立的用户故事，每个故事应遵循INVEST原则（独立、可协商、有价值、可估算、小、可测试）。

每个用户故事应包含：
- title: 简洁的标题
- description: 详细描述，使用"作为[角色]，我想要[功能]，以便[原因]"的格式
- acceptance_criteria: 清晰的验收标准列表，使用可测试的语言
- priority: 优先级（High, Medium, Low）
- estimated_points: 估计的故事点数（1-13）
- risk_level: 风险级别（Low, Medium, High）
- dependencies: 依赖的其他用户故事或组件

请以JSON格式返回生成的用户故事列表，字段名为"user_stories"。

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API生成用户故事
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位敏捷开发专家，擅长将需求转换为高质量的用户故事。请遵循敏捷开发的最佳实践。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("user_stories", [])
            
        except Exception as e:
            print(f"生成用户故事失败: {e}")
            return []
    
    @staticmethod
    def generate_sprint_backlog(user_stories: List[Dict[str, Any]], sprint_capacity: int = 20, team_composition: str = "") -> Dict[str, Any]:
        """
        基于用户故事和团队容量生成冲刺待办事项
        """
        try:
            # 将用户故事列表转换为JSON字符串
            user_stories_json = json.dumps(user_stories, indent=2)
            
            # 构建提示词
            prompt = f"""
请基于以下用户故事列表、团队容量和团队组成信息，生成一个优化的冲刺待办事项。

团队容量: {sprint_capacity} 故事点

团队组成: {team_composition or "标准开发团队"}

用户故事列表:
{user_stories_json}

请执行以下任务：
1. 按照优先级、依赖关系和风险级别对用户故事进行排序
2. 选择能够在给定容量内完成的用户故事
3. 为每个用户故事生成具体的任务分解
4. 估计每个任务的工作量
5. 考虑团队组成和专长进行合理分配

请以JSON格式返回冲刺待办事项，包含以下字段：
- selected_stories: 选中的用户故事列表
- total_points: 总故事点数
- tasks: 任务列表，每个任务包含：
  - id: 任务ID
  - description: 任务描述
  - story_id: 所属用户故事ID（如果有）
  - estimated_hours: 估计工时
  - assigned_role: 建议分配的角色
  - priority: 优先级（High, Medium, Low）
- sprint_goals: 冲刺目标
- potential_risks: 潜在风险
- sprint_summary: 冲刺摘要

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API生成冲刺待办事项
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位敏捷开发教练和Scrum大师，擅长规划和优化冲刺。请基于提供的信息创建一个平衡且可行的冲刺计划。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"生成冲刺待办事项失败: {e}")
            return {
                "selected_stories": [],
                "total_points": 0,
                "tasks": [],
                "sprint_goals": [],
                "potential_risks": [],
                "sprint_summary": f"生成失败: {str(e)}"
            }
    
    @staticmethod
    def validate_user_story(user_story: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证用户故事是否符合INVEST原则
        """
        try:
            # 将用户故事转换为JSON字符串
            user_story_json = json.dumps(user_story, indent=2)
            
            # 构建提示词
            prompt = f"""
请作为一位敏捷开发专家，验证以下用户故事是否符合INVEST原则（独立、可协商、有价值、可估算、小、可测试）。

用户故事:
{user_story_json}

请分析该用户故事是否符合每个INVEST原则，并提供具体的改进建议。

请以JSON格式返回分析结果，包含以下字段：
- independent: 是否独立
- negotiable: 是否可协商
- valuable: 是否有价值
- estimable: 是否可估算
- small: 是否足够小
- testable: 是否可测试
- overall_score: 总体评分（0-100）
- strengths: 优势
- weaknesses: 劣势
- improvement_suggestions: 改进建议

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API验证用户故事
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位敏捷开发专家，擅长评估用户故事的质量和遵循最佳实践。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"验证用户故事失败: {e}")
            return {
                "independent": False,
                "negotiable": False,
                "valuable": False,
                "estimable": False,
                "small": False,
                "testable": False,
                "overall_score": 0,
                "strengths": [],
                "weaknesses": [],
                "improvement_suggestions": [f"验证失败: {str(e)}"]
            }

# 实用工具函数
def generate_requirement_report(analysis_result: Dict[str, Any], output_file: str) -> None:
    """
    生成需求分析报告
    """
    # 生成HTML报告
    html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>需求分析报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 10px 20px;
            border-radius: 8px 8px 0 0;
        }}
        .content {{
            background-color: white;
            padding: 20px;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .score-container {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }}
        .score-card {{
            text-align: center;
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: 8px;
            width: 45%;
        }}
        .score {{
            font-size: 48px;
            font-weight: bold;
            color: #1976d2;
        }}
        .section {{
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }}
        .user-story {{
            background-color: #f5f5f5;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #2196f3;
            border-radius: 4px;
        }}
        .priority-high {{
            border-left-color: #f44336;
        }}
        .priority-medium {{
            border-left-color: #ff9800;
        }}
        .priority-low {{
            border-left-color: #4caf50;
        }}
        .ac-list {{
            list-style-type: none;
            padding: 0;
        }}
        .ac-list li {{
            padding: 5px 0;
            padding-left: 20px;
            position: relative;
        }}
        .ac-list li:before {{
            content: '✓';
            position: absolute;
            left: 0;
            color: #4caf50;
        }}
        ul, ol {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>需求分析报告</h1>
    </div>
    
    <div class="content">
        <div class="section">
            <h2>文档信息</h2>
            <p>文档路径: {analysis_result.get('document_path', 'Unknown')}</p>
        </div>
        
        <div class="section">
            <h2>总体摘要</h2>
            <p>{analysis_result.get('executive_summary', '无摘要信息')}</p>
        </div>
        
        <div class="section">
            <h2>评分概览</h2>
            <div class="score-container">
                <div class="score-card">
                    <div class="score">{analysis_result.get('clarity_score', 0)}</div>
                    <div>清晰度评分</div>
                </div>
                <div class="score-card">
                    <div class="score">{analysis_result.get('completeness_score', 0)}</div>
                    <div>完整性评分</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>发现的歧义</h2>
            {_generate_list_html(analysis_result.get('identified_ambiguities', []), '未发现歧义')}
        </div>
        
        <div class="section">
            <h2>缺失的需求</h2>
            {_generate_list_html(analysis_result.get('missing_requirements', []), '未发现明显缺失的需求')}
        </div>
        
        <div class="section">
            <h2>风险和挑战</h2>
            {_generate_list_html(analysis_result.get('risks_and_challenges', []), '未发现明显风险')}
        </div>
        
        <div class="section">
            <h2>改进建议</h2>
            {_generate_list_html(analysis_result.get('improvement_suggestions', []), '无改进建议')}
        </div>
        
        <div class="section">
            <h2>建议的用户故事</h2>
            {_generate_user_stories_html(analysis_result.get('suggested_user_stories', []))}
        </div>
    </div>
</body>
</html>
"""
    
    # 写入报告文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"需求分析报告已生成: {output_file}")

def _generate_list_html(items: List[str], empty_message: str = "") -> str:
    """
    生成列表的HTML
    """
    if not items:
        return f"<p>{empty_message}</p>"
    
    html = "<ul>"
    for item in items:
        html += f"<li>{item}</li>"
    html += "</ul>"
    
    return html

def _generate_user_stories_html(user_stories: List[Dict[str, Any]]) -> str:
    """
    生成用户故事的HTML
    """
    if not user_stories:
        return "<p>未生成用户故事</p>"
    
    html = ""
    for story in user_stories:
        priority_class = f"priority-{story.get('priority', 'medium').lower()}"
        html += f"""
        <div class="user-story {priority_class}">
            <h3>{story.get('title', 'Untitled')}</h3>
            <p><strong>描述:</strong> {story.get('description', 'No description')}</p>
            <p><strong>优先级:</strong> {story.get('priority', 'Medium')}</p>
            <p><strong>验收标准:</strong></p>
            <ul class="ac-list">
                {''.join([f"<li>{criteria}</li>" for criteria in story.get('acceptance_criteria', [])])}
            </ul>
        </div>
        """
    
    return html

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用AI进行智能需求分析")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 分析需求文档命令
    analyze_parser = subparsers.add_parser("analyze", help="分析需求文档")
    analyze_parser.add_argument("document", help="需求文档路径")
    analyze_parser.add_argument("--output", help="输出报告文件路径")
    
    # 生成用户故事命令
    stories_parser = subparsers.add_parser("stories", help="生成用户故事")
    stories_parser.add_argument("--file", help="包含需求文本的文件路径")
    stories_parser.add_argument("--text", help="需求文本")
    stories_parser.add_argument("--context", help="项目上下文信息")
    stories_parser.add_argument("--output", help="输出文件路径")
    
    # 生成冲刺待办事项命令
    sprint_parser = subparsers.add_parser("sprint", help="生成冲刺待办事项")
    sprint_parser.add_argument("stories_file", help="用户故事JSON文件路径")
    sprint_parser.add_argument("--capacity", type=int, default=20, help="团队容量（故事点）")
    sprint_parser.add_argument("--team", help="团队组成信息")
    sprint_parser.add_argument("--output", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 检查OpenAI API密钥
    if not openai.api_key:
        print("错误: 未设置OPENAI_API_KEY环境变量")
        sys.exit(1)
    
    if args.command == "analyze":
        # 分析需求文档
        result = AIRequirementAnalyzer.analyze_requirement_document(args.document)
        
        # 生成报告
        if args.output:
            generate_requirement_report(result, args.output)
        else:
            # 打印结果
            print(f"\n清晰度评分: {result['clarity_score']}/100")
            print(f"完整性评分: {result['completeness_score']}/100")
            print(f"\n总体摘要: {result['executive_summary']}")
            
            if result['identified_ambiguities']:
                print(f"\n发现的歧义 ({len(result['identified_ambiguities'])}):")
                for ambiguity in result['identified_ambiguities']:
                    print(f"- {ambiguity}")
            
            if result['missing_requirements']:
                print(f"\n缺失的需求 ({len(result['missing_requirements'])}):")
                for req in result['missing_requirements']:
                    print(f"- {req}")
            
            if result['suggested_user_stories']:
                print(f"\n建议的用户故事 ({len(result['suggested_user_stories'])}):")
                for story in result['suggested_user_stories']:
                    print(f"\n- {story['title']} (优先级: {story['priority']})")
                    print(f"  描述: {story['description']}")
                    print("  验收标准:")
                    for criteria in story['acceptance_criteria']:
                        print(f"    - {criteria}")
    
    elif args.command == "stories":
        # 生成用户故事
        requirement_text = ""
        
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                requirement_text = f.read()
        elif args.text:
            requirement_text = args.text
        else:
            print("错误: 必须提供 --file 或 --text 参数")
            sys.exit(1)
        
        user_stories = AIRequirementAnalyzer.generate_user_stories(requirement_text, args.context)
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(user_stories, f, indent=2, ensure_ascii=False)
            print(f"用户故事已生成并保存至: {args.output}")
        else:
            print(f"\n生成的用户故事 ({len(user_stories)}):")
            for i, story in enumerate(user_stories, 1):
                print(f"\n{i}. {story['title']}")
                print(f"   描述: {story['description']}")
                print(f"   优先级: {story['priority']}")
                print(f"   故事点: {story.get('estimated_points', 'N/A')}")
                print("   验收标准:")
                for criteria in story.get('acceptance_criteria', []):
                    print(f"     - {criteria}")
    
    elif args.command == "sprint":
        # 生成冲刺待办事项
        try:
            with open(args.stories_file, 'r', encoding='utf-8') as f:
                user_stories = json.load(f)
        except Exception as e:
            print(f"读取用户故事文件失败: {e}")
            sys.exit(1)
        
        sprint_backlog = AIRequirementAnalyzer.generate_sprint_backlog(user_stories, args.capacity, args.team)
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(sprint_backlog, f, indent=2, ensure_ascii=False)
            print(f"冲刺待办事项已生成并保存至: {args.output}")
        else:
            print(f"\n冲刺摘要: {sprint_backlog['sprint_summary']}")
            print(f"总故事点数: {sprint_backlog['total_points']}/{args.capacity}")
            print(f"选中的故事数: {len(sprint_backlog['selected_stories'])}")
            print(f"生成的任务数: {len(sprint_backlog['tasks'])}")
            
            print(f"\n冲刺目标:")
            for goal in sprint_backlog['sprint_goals']:
                print(f"- {goal}")
            
            print(f"\n潜在风险:")
            for risk in sprint_backlog['potential_risks']:
                print(f"- {risk}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### 智能用户故事管理

以下是一个使用AI辅助用户故事管理的脚本：

```python
def validate_user_stories_batch(user_stories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量验证用户故事
    """
    results = []
    for i, story in enumerate(user_stories, 1):
        print(f"验证用户故事 {i}/{len(user_stories)}: {story.get('title', 'Untitled')}")
        result = AIRequirementAnalyzer.validate_user_story(story)
        # 将验证结果与原始故事合并
        merged_result = story.copy()
        merged_result['validation'] = result
        results.append(merged_result)
    return results

def optimize_user_story(user_story: Dict[str, Any]) -> Dict[str, Any]:
    """
    优化用户故事
    """
    try:
        # 将用户故事转换为JSON字符串
        user_story_json = json.dumps(user_story, indent=2)
        
        # 构建提示词
        prompt = f"""
请作为一位敏捷开发专家，优化以下用户故事，使其更好地符合INVEST原则。

原始用户故事:
{user_story_json}

请提供一个优化后的用户故事，保持原有的核心需求不变，但改进其结构、清晰度和可执行性。

优化后的用户故事应包含：
- title: 简洁明了的标题
- description: 详细描述，使用"作为[角色]，我想要[功能]，以便[原因]"的格式
- acceptance_criteria: 明确、可测试的验收标准列表
- priority: 优先级（High, Medium, Low）
- estimated_points: 估计的故事点数（1-13）
- risk_level: 风险级别（Low, Medium, High）
- notes: 优化说明和建议

请以JSON格式返回优化后的用户故事。

请确保JSON格式正确，不要包含JSON之外的内容。
"""
        
        # 调用AI API优化用户故事
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一位敏捷开发专家，擅长优化用户故事。请提供具体、实用的改进。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"优化用户故事失败: {e}")
        return user_story

def generate_user_story_card(user_story: Dict[str, Any], output_format: str = "markdown") -> str:
    """
    生成用户故事卡片
    """
    if output_format == "markdown":
        # 生成Markdown格式
        card = f"""
# {user_story.get('title', 'Untitled')}

## 描述
{user_story.get('description', 'No description')}

## 验收标准
{''.join([f"- [ ] {criteria}\n" for criteria in user_story.get('acceptance_criteria', [])])}

## 详情
- **优先级**: {user_story.get('priority', 'Medium')}
- **故事点**: {user_story.get('estimated_points', 'N/A')}
- **风险级别**: {user_story.get('risk_level', 'Medium')}
"""
        
        if user_story.get('dependencies', []):
            card += f"- **依赖**: {', '.join(user_story['dependencies'])}\n"
        
        return card.strip()
    
    elif output_format == "html":
        # 生成HTML格式
        priority_color_map = {
            "High": "#f44336",
            "Medium": "#ff9800",
            "Low": "#4caf50"
        }
        priority_color = priority_color_map.get(user_story.get('priority', 'Medium'), "#ff9800")
        
        card = f"""
<div style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px; width: 300px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h2 style="margin-top: 0; color: {priority_color};">{user_story.get('title', 'Untitled')}</h2>
    <div style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
        <p>{user_story.get('description', 'No description')}</p>
    </div>
    <h3 style="margin-top: 15px; margin-bottom: 5px;">验收标准</h3>
    <ul style="margin-top: 0;">
        {''.join([f"<li><input type='checkbox'> {criteria}</li>" for criteria in user_story.get('acceptance_criteria', [])])}
    </ul>
    <div style="margin-top: 15px; font-size: 12px; color: #666;">
        <p>优先级: <strong>{user_story.get('priority', 'Medium')}</strong></p>
        <p>故事点: {user_story.get('estimated_points', 'N/A')}</p>
        <p>风险级别: {user_story.get('risk_level', 'Medium')}</p>
    </div>
</div>
"""
        
        return card
    
    else:
        return "不支持的输出格式"
```

## 6.6.2 智能构建与部署自动化

AI可以帮助优化和自动化DevOps流程中的构建和部署步骤，提高效率和可靠性。

### 智能构建配置生成

以下是一个使用AI生成和优化构建配置的脚本示例：

```python
#!/usr/bin/env python3
# ai_build_config_generator.py - 使用AI生成和优化构建配置

import os
import sys
import openai
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# 配置OpenAI API密钥
openai.api_key = os.environ.get("OPENAI_API_KEY")

class AIBuildConfigGenerator:
    """
    智能构建配置生成器，使用AI生成和优化各种构建工具的配置文件
    """
    
    @staticmethod
    def generate_build_config(project_type: str, framework: str = None, 
                            build_system: str = None, features: list = None) -> Dict[str, Any]:
        """
        生成构建配置文件
        """
        try:
            # 确定合适的构建系统
            if not build_system:
                build_system_map = {
                    "javascript": "webpack" if framework in ["react", "vue", "angular"] else "rollup",
                    "typescript": "webpack",
                    "python": "pytest",
                    "java": "maven",
                    "go": "go build",
                    "rust": "cargo",
                    "dotnet": "msbuild",
                    "php": "composer",
                    "ruby": "rake"
                }
                build_system = build_system_map.get(project_type.lower(), "custom")
            
            # 构建提示词
            prompt = f"""
请生成一个针对{project_type}项目{"，使用{framework}框架" if framework else ""}的{build_system}构建配置。

以下是需要包含的功能和要求：
{''.join([f"- {feature}\n" for feature in (features or [])])}

请提供以下内容：
1. 配置文件的完整内容
2. 配置文件的文件名
3. 配置的简要说明和关键部分的解释
4. 如何使用该配置的简单指南

请以JSON格式返回，包含以下字段：
- config_content: 配置文件的完整内容
- config_filename: 配置文件的推荐文件名
- explanation: 配置说明
- usage_guide: 使用指南

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API生成配置
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位DevOps专家，擅长各种构建工具和配置。请提供准确、实用的配置文件。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"生成构建配置失败: {e}")
            return {
                "config_content": "",
                "config_filename": "",
                "explanation": f"生成失败: {str(e)}",
                "usage_guide": ""
            }
    
    @staticmethod
    def optimize_build_config(config_file: str, optimization_goals: list = None) -> Dict[str, Any]:
        """
        优化现有构建配置文件
        """
        try:
            # 读取配置文件内容
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 获取文件扩展名
            file_ext = os.path.splitext(config_file)[1].lower()
            
            # 确定配置类型
            config_type_map = {
                '.json': 'JSON',
                '.js': 'JavaScript',
                '.yml': 'YAML',
                '.yaml': 'YAML',
                '.xml': 'XML',
                '.toml': 'TOML',
                '.py': 'Python',
                '.sh': 'Shell script',
                '.bat': 'Batch script',
                '.cmd': 'Batch script',
                '.gradle': 'Gradle',
                '.pom': 'Maven POM',
                '.csproj': 'MSBuild',
                '.go': 'Go',
                '.rs': 'Rust',
                'Dockerfile': 'Dockerfile',
                'docker-compose.yml': 'Docker Compose'
            }
            config_type = config_type_map.get(file_ext, '通用配置')
            
            # 构建提示词
            prompt = f"""
请优化以下{config_type}格式的构建配置文件。

配置文件路径: {config_file}

原始配置内容:
```
{content}
```

优化目标:
{''.join([f"- {goal}\n" for goal in (optimization_goals or [])])}

请提供以下内容：
1. 优化后的配置文件完整内容
2. 对优化部分的详细说明和理由
3. 优化后可能带来的好处

请以JSON格式返回，包含以下字段：
- optimized_content: 优化后的配置文件内容
- optimization_explanations: 优化说明列表
- benefits: 预期好处列表

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API优化配置
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位DevOps和性能优化专家，擅长优化各种构建配置。请提供具体、可量化的改进。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 添加文件信息
            result['original_file'] = config_file
            
            return result
            
        except Exception as e:
            print(f"优化构建配置失败 {config_file}: {e}")
            return {
                "original_file": config_file,
                "optimized_content": "",
                "optimization_explanations": [f"优化失败: {str(e)}"],
                "benefits": []
            }
    
    @staticmethod
    def generate_ci_cd_pipeline(pipeline_type: str, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成CI/CD流水线配置
        """
        try:
            # 构建提示词
            prompt = f"""
请为以下项目生成{"GitHub Actions" if pipeline_type.lower() == "github" else "GitLab CI" if pipeline_type.lower() == "gitlab" else "Jenkins"} CI/CD流水线配置。

项目信息:
{json.dumps(project_info, indent=2)}

请提供以下内容：
1. 完整的CI/CD配置文件内容
2. 配置文件的推荐文件名
3. 对配置的解释和关键点说明
4. 使用建议和最佳实践

请以JSON格式返回，包含以下字段：
- pipeline_content: CI/CD配置文件内容
- pipeline_filename: 推荐的文件名
- explanation: 配置说明
- best_practices: 最佳实践建议

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API生成CI/CD流水线
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位CI/CD专家，擅长配置各种CI/CD系统。请提供功能完整、遵循最佳实践的流水线配置。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"生成CI/CD流水线失败: {e}")
            return {
                "pipeline_content": "",
                "pipeline_filename": "",
                "explanation": f"生成失败: {str(e)}",
                "best_practices": []
            }
    
    @staticmethod
    def analyze_build_performance(log_file: str) -> Dict[str, Any]:
        """
        分析构建日志，提供性能优化建议
        """
        try:
            # 读取构建日志
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # 限制日志大小
            max_log_size = 5000
            if len(log_content) > max_log_size:
                log_content = log_content[-max_log_size:]  # 取最后一部分日志
                log_content = "[日志过长，只显示最后部分]\n" + log_content
            
            # 构建提示词
            prompt = f"""
请分析以下构建日志，识别性能瓶颈和优化机会。

日志文件: {log_file}

日志内容:
```
{log_content}
```

请提供以下分析：
1. 构建总时间和各阶段耗时
2. 性能瓶颈和问题点
3. 具体的优化建议
4. 预期的性能提升

请以JSON格式返回分析结果，包含以下字段：
- build_summary: 构建摘要（总时间、主要阶段等）
- bottlenecks: 性能瓶颈列表，每个包含：
  - stage: 阶段
  - duration: 耗时（如果能识别）
  - description: 问题描述
  - severity: 严重程度（High, Medium, Low）
- optimization_suggestions: 优化建议列表，每个包含：
  - description: 建议描述
  - expected_impact: 预期影响
  - implementation_steps: 实施步骤
- overall_assessment: 总体评估和建议

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API分析构建性能
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位构建系统专家，擅长分析构建性能并提供优化建议。请基于日志内容提供具体、可行的改进方案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"分析构建性能失败: {e}")
            return {
                "build_summary": "",
                "bottlenecks": [],
                "optimization_suggestions": [],
                "overall_assessment": f"分析失败: {str(e)}"
            }

# 实用工具函数
def save_config(config_content: str, filename: str, output_dir: str = None) -> None:
    """
    保存配置文件
    """
    # 确定输出目录
    if not output_dir:
        output_dir = os.getcwd()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建完整文件路径
    file_path = os.path.join(output_dir, filename)
    
    # 保存文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"配置文件已保存至: {file_path}")

def generate_performance_report(analysis_result: Dict[str, Any], output_file: str) -> None:
    """
    生成构建性能分析报告
    """
    # 生成HTML报告
    html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>构建性能分析报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 10px 20px;
            border-radius: 8px 8px 0 0;
        }}
        .content {{
            background-color: white;
            padding: 20px;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section {{
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }}
        .bottleneck-item {{
            background-color: #ffebee;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #f44336;
            border-radius: 4px;
        }}
        .severity-high {{
            border-left-color: #f44336;
        }}
        .severity-medium {{
            border-left-color: #ff9800;
        }}
        .severity-low {{
            border-left-color: #4caf50;
        }}
        .suggestion-item {{
            background-color: #e8f5e9;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #4caf50;
            border-radius: 4px;
        }}
        .expected-impact-high {{
            border-left-color: #4caf50;
        }}
        .expected-impact-medium {{
            border-left-color: #ff9800;
        }}
        .expected-impact-low {{
            border-left-color: #2196f3;
        }}
        ul, ol {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>构建性能分析报告</h1>
    </div>
    
    <div class="content">
        <div class="section">
            <h2>构建摘要</h2>
            <pre>{analysis_result.get('build_summary', '无摘要信息')}</pre>
        </div>
        
        <div class="section">
            <h2>性能瓶颈</h2>
            {_generate_bottlenecks_html(analysis_result.get('bottlenecks', []))}
        </div>
        
        <div class="section">
            <h2>优化建议</h2>
            {_generate_suggestions_html(analysis_result.get('optimization_suggestions', []))}
        </div>
        
        <div class="section">
            <h2>总体评估</h2>
            <p>{analysis_result.get('overall_assessment', '无评估信息')}</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 写入报告文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"构建性能分析报告已生成: {output_file}")

def _generate_bottlenecks_html(bottlenecks: list) -> str:
    """
    生成性能瓶颈的HTML
    """
    if not bottlenecks:
        return "<p>未发现明显性能瓶颈</p>"
    
    html = ""
    for bottleneck in bottlenecks:
        severity_class = f"severity-{bottleneck.get('severity', 'medium').lower()}"
        html += f"""
        <div class="bottleneck-item {severity_class}">
            <h3>{bottleneck.get('stage', '未知阶段')}</h3>
            <p><strong>耗时:</strong> {bottleneck.get('duration', '未知')}</p>
            <p><strong>严重程度:</strong> {bottleneck.get('severity', '未知')}</p>
            <p><strong>问题描述:</strong> {bottleneck.get('description', '无描述')}</p>
        </div>
        """
    
    return html

def _generate_suggestions_html(suggestions: list) -> str:
    """
    生成优化建议的HTML
    """
    if not suggestions:
        return "<p>无优化建议</p>"
    
    html = ""
    for suggestion in suggestions:
        # 根据预期影响确定样式
        impact_level = suggestion.get('expected_impact', 'Medium')
        impact_class = f"expected-impact-{impact_level.lower()}"
        html += f"""
        <div class="suggestion-item {impact_class}">
            <h3>{suggestion.get('description', '未命名建议')}</h3>
            <p><strong>预期影响:</strong> {impact_level}</p>
            <p><strong>实施步骤:</strong></p>
            <ol>
                {''.join([f"<li>{step}</li>" for step in suggestion.get('implementation_steps', [])])}
            </ol>
        </div>
        """
    
    return html

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用AI生成和优化构建配置")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 生成构建配置命令
    gen_config_parser = subparsers.add_parser("generate", help="生成构建配置")
    gen_config_parser.add_argument("--type", required=True, help="项目类型")
    gen_config_parser.add_argument("--framework", help="使用的框架")
    gen_config_parser.add_argument("--build-system", help="构建系统")
    gen_config_parser.add_argument("--feature", action="append", dest="features", help="需要包含的功能")
    gen_config_parser.add_argument("--output", help="输出目录")
    
    # 优化构建配置命令
    optimize_parser = subparsers.add_parser("optimize", help="优化构建配置")
    optimize_parser.add_argument("config_file", help="配置文件路径")
    optimize_parser.add_argument("--goal", action="append", dest="goals", help="优化目标")
    optimize_parser.add_argument("--output", help="输出目录")
    
    # 生成CI/CD流水线命令
    ci_cd_parser = subparsers.add_parser("pipeline", help="生成CI/CD流水线配置")
    ci_cd_parser.add_argument("--type", required=True, choices=["github", "gitlab", "jenkins"], help="CI/CD系统类型")
    ci_cd_parser.add_argument("--project-info", required=True, help="项目信息JSON文件路径")
    ci_cd_parser.add_argument("--output", help="输出目录")
    
    # 分析构建性能命令
    perf_parser = subparsers.add_parser("analyze", help="分析构建性能")
    perf_parser.add_argument("log_file", help="构建日志文件路径")
    perf_parser.add_argument("--report", help="输出报告文件路径")
    
    args = parser.parse_args()
    
    # 检查OpenAI API密钥
    if not openai.api_key:
        print("错误: 未设置OPENAI_API_KEY环境变量")
        sys.exit(1)
    
    if args.command == "generate":
        # 生成构建配置
        result = AIBuildConfigGenerator.generate_build_config(
            args.type,
            args.framework,
            args.build_system,
            args.features
        )
        
        # 保存配置文件
        if result.get("config_content") and result.get("config_filename"):
            save_config(result["config_content"], result["config_filename"], args.output)
            print(f"\n配置说明: {result.get('explanation', '')}")
            print(f"\n使用指南: {result.get('usage_guide', '')}")
        else:
            print("生成配置失败")
    
    elif args.command == "optimize":
        # 优化构建配置
        result = AIBuildConfigGenerator.optimize_build_config(args.config_file, args.goals)
        
        # 保存优化后的配置
        if result.get("optimized_content"):
            # 使用与原文件相同的名称，但添加.optimized后缀
            original_dir = os.path.dirname(args.config_file)
            original_name = os.path.basename(args.config_file)
            name, ext = os.path.splitext(original_name)
            optimized_filename = f"{name}.optimized{ext}"
            
            # 确定输出目录
            output_dir = args.output if args.output else original_dir
            
            save_config(result["optimized_content"], optimized_filename, output_dir)
            
            print("\n优化说明:")
            for explanation in result.get("optimization_explanations", []):
                print(f"- {explanation}")
            
            print("\n预期好处:")
            for benefit in result.get("benefits", []):
                print(f"- {benefit}")
        else:
            print("优化配置失败")
    
    elif args.command == "pipeline":
        # 生成CI/CD流水线配置
        try:
            with open(args.project_info, 'r', encoding='utf-8') as f:
                project_info = json.load(f)
        except Exception as e:
            print(f"读取项目信息文件失败: {e}")
            sys.exit(1)
        
        result = AIBuildConfigGenerator.generate_ci_cd_pipeline(args.type, project_info)
        
        # 保存配置文件
        if result.get("pipeline_content") and result.get("pipeline_filename"):
            save_config(result["pipeline_content"], result["pipeline_filename"], args.output)
            print(f"\n配置说明: {result.get('explanation', '')}")
            
            print("\n最佳实践:")
            for practice in result.get("best_practices", []):
                print(f"- {practice}")
        else:
            print("生成CI/CD流水线配置失败")
    
    elif args.command == "analyze":
        # 分析构建性能
        result = AIBuildConfigGenerator.analyze_build_performance(args.log_file)
        
        # 生成报告
        if args.report:
            generate_performance_report(result, args.report)
        else:
            # 打印结果
            print(f"\n构建摘要: {result.get('build_summary', '无摘要')}")
            
            if result.get('bottlenecks', []):
                print(f"\n性能瓶颈 ({len(result['bottlenecks'])}):")
                for bottleneck in result['bottlenecks']:
                    print(f"- [{bottleneck.get('severity', 'N/A')}] {bottleneck.get('stage', 'Unknown')}: {bottleneck.get('description', '')}")
            
            if result.get('optimization_suggestions', []):
                print(f"\n优化建议 ({len(result['optimization_suggestions'])}):")
                for suggestion in result['optimization_suggestions']:
                    print(f"- {suggestion.get('description', '')}")
                    print(f"  预期影响: {suggestion.get('expected_impact', 'N/A')}")
                    print("  实施步骤:")
                    for step in suggestion.get('implementation_steps', []):
                        print(f"    - {step}")
            
            print(f"\n总体评估: {result.get('overall_assessment', '无评估')}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### 智能部署策略生成

以下是一个使用AI生成部署策略的脚本：

```python
def generate_deployment_strategy(deployment_type: str, environment_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成部署策略
    """
    try:
        # 构建提示词
        prompt = f"""
请为以下环境生成{deployment_type}部署策略。

环境信息:
{json.dumps(environment_info, indent=2)}

请提供以下内容：
1. 详细的部署策略说明
2. 部署流程和步骤
3. 回滚策略
4. 监控和验证方法
5. 最佳实践和建议

请以JSON格式返回，包含以下字段：
- strategy_name: 策略名称
- strategy_description: 策略描述
- deployment_steps: 部署步骤列表
- rollback_strategy: 回滚策略
- monitoring_verification: 监控和验证方法
- best_practices: 最佳实践列表
- tools_recommendations: 推荐的工具和配置

请确保JSON格式正确，不要包含JSON之外的内容。
"""
        
        # 调用AI API生成部署策略
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一位DevOps和部署专家，擅长设计各种部署策略。请提供详细、实用的部署方案。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"生成部署策略失败: {e}")
        return {
            "strategy_name": "",
            "strategy_description": f"生成失败: {str(e)}",
            "deployment_steps": [],
            "rollback_strategy": "",
            "monitoring_verification": "",
            "best_practices": [],
            "tools_recommendations": []
        }

def generate_infrastructure_as_code(infra_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成基础设施即代码配置
    """
    try:
        # 构建提示词
        prompt = f"""
请生成{infra_type}格式的基础设施即代码配置，满足以下需求：

需求:
{json.dumps(requirements, indent=2)}

请提供以下内容：
1. 完整的基础设施即代码配置
2. 配置说明和关键部分解释
3. 使用和部署指南
4. 最佳实践建议

请以JSON格式返回，包含以下字段：
- code_content: 基础设施即代码配置内容
- code_filename: 推荐的文件名
- explanation: 配置说明
- deployment_guide: 部署指南
- best_practices: 最佳实践建议

请确保JSON格式正确，不要包含JSON之外的内容。
"""
        
        # 调用AI API生成基础设施即代码
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一位云基础设施专家，擅长编写Terraform、CloudFormation等基础设施即代码。请提供符合最佳实践的配置。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"生成基础设施即代码失败: {e}")
        return {
            "code_content": "",
            "code_filename": "",
            "explanation": f"生成失败: {str(e)}",
            "deployment_guide": "",
            "best_practices": []
        }
```

## 6.6.3 DevOps流程优化与监控

AI可以帮助优化DevOps流程，并提供智能监控和预警功能。

### 智能监控与预警系统

以下是一个使用AI进行智能监控与预警的脚本示例：

```python
#!/usr/bin/env python3
# ai_monitoring_system.py - 使用AI进行智能监控与预警

import os
import sys
import openai
import json
import argparse
import time
import random
from datetime import datetime
from typing import List, Dict, Any

# 配置OpenAI API密钥
openai.api_key = os.environ.get("OPENAI_API_KEY")

class AIMonitoringSystem:
    """
    智能监控系统，使用AI分析监控数据并提供预警和优化建议
    """
    
    @staticmethod
    def analyze_logs(log_file: str, pattern: str = None, time_range: str = None) -> Dict[str, Any]:
        """
        分析日志文件，识别异常模式和潜在问题
        """
        try:
            # 读取日志文件
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # 限制日志大小
            max_log_size = 5000
            if len(log_content) > max_log_size:
                log_content = log_content[-max_log_size:]  # 取最后一部分日志
                log_content = "[日志过长，只显示最后部分]\n" + log_content
            
            # 构建提示词
            prompt = f"""
请作为一位运维专家，分析以下日志文件，识别异常模式、错误和潜在问题。

日志文件: {log_file}
{"搜索模式: " + pattern + "\n" if pattern else ""}
{"时间范围: " + time_range + "\n" if time_range else ""}

日志内容:
```
{log_content}
```

请提供以下分析：
1. 识别的错误和异常
2. 模式识别和趋势分析
3. 潜在问题和风险
4. 改进建议

请以JSON格式返回分析结果，包含以下字段：
- errors_detected: 检测到的错误和异常列表，每个包含：
  - type: 错误类型
  - count: 出现次数
  - sample: 示例
  - severity: 严重程度（Critical, High, Medium, Low）
  - impact: 潜在影响
- patterns_identified: 识别的模式和趋势列表
- potential_issues: 潜在问题和风险列表
- improvement_suggestions: 改进建议列表
- overall_assessment: 总体评估

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API分析日志
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位资深的运维专家，擅长分析各种系统日志并识别潜在问题。请提供详细、准确的分析。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"分析日志失败 {log_file}: {e}")
            return {
                "errors_detected": [],
                "patterns_identified": [],
                "potential_issues": [],
                "improvement_suggestions": [],
                "overall_assessment": f"分析失败: {str(e)}"
            }
    
    @staticmethod
    def analyze_metrics(metrics_data: Dict[str, Any], threshold_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        分析性能指标，识别异常和性能问题
        """
        try:
            # 构建提示词
            prompt = f"""
请作为一位性能分析专家，分析以下系统性能指标数据，识别异常模式、性能瓶颈和潜在问题。

性能指标数据:
{json.dumps(metrics_data, indent=2)}

{"阈值规则: " + json.dumps(threshold_rules, indent=2) + "\n" if threshold_rules else ""}

请提供以下分析：
1. 异常指标识别
2. 性能瓶颈分析
3. 潜在问题和风险评估
4. 性能优化建议

请以JSON格式返回分析结果，包含以下字段：
- anomalies: 检测到的异常指标列表，每个包含：
  - metric_name: 指标名称
  - value: 实际值
  - expected_range: 预期范围
  - severity: 严重程度（Critical, High, Medium, Low）
  - description: 异常描述
- bottlenecks: 性能瓶颈列表
- potential_issues: 潜在问题和风险列表
- optimization_suggestions: 性能优化建议列表
- overall_performance_score: 总体性能评分（0-100）
- trend_analysis: 趋势分析

请确保JSON格式正确，不要包含JSON之外的内容。
"""
            
            # 调用AI API分析性能指标
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位性能分析专家，擅长分析系统指标数据并识别性能问题。请提供详细、专业的分析。"},
                    {"role": "user", "content": prompt}