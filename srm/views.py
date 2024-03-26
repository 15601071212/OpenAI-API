from django.shortcuts import render,get_object_or_404,redirect
from django.views.decorators.cache import cache_page
from django.core.cache import cache

import re

import time
from srm.models import KeywordPost
from srm.models import KeywordList
from srm.models import ScriptPost
from srm.models import ScriptList
from srm.models import KeywordSearch
from srm.models import KeywordDoc
from srm.models import ConfigSearch
from srm.models import ResultFeedback

from django.http import JsonResponse
from django.views import View
from django.core import serializers
from django.http import HttpResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib import messages
from django.http import HttpResponseRedirect
import json
import socket
from itertools import zip_longest
from django.db.models import Q
from functools import reduce
import operator

from pygments import highlight
from pygments.lexers import RobotFrameworkLexer
from pygments.formatters import HtmlFormatter
import os
import datetime
from datetime import date
#from datetime import datetime
from collections import Counter
import random
import requests
import sys
sys.path.append('/opt/tester/git_repo/AutoCenter/lib') 
from ztesw.misc.ConfigTransformers.main import ConfigTrans

#import multiprocessing

from rest_framework.decorators import api_view 
from rest_framework.response import Response
from .forms import LoadingForm
from celery import shared_task
#from celery import task 
from celery import Celery
from celery.result import AsyncResult

app = Celery('srm', broker='redis://localhost:6379', backend='redis://localhost:6379')

from srm.agent.mylangchain import ( 
                         get_multi_samples,
                         )

from srm.agent.my_openai import (
                         get_multi_scripts_examples_chat,
                         get_keyword_chat)

from pyecharts import options as opts
from pyecharts.charts import Bar, Gauge, Pie, Page, Funnel, Geo, Scatter3D, Line
from bs4 import BeautifulSoup
from multiprocessing import dummy as multiprocessing
from functools import partial

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from ztesw.misc.ConfigTransformers.auto_learn.run import AutoLearn

from langchain.chains import LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from srm.agent import Example
from srm.agent.local_llm import JuLLMChatChat, OpenAIModel

from srm.agent.utility import timeout, is_valid_json_with_key
import tiktoken
from loguru import logger

from functools import wraps
import queue
from queue import Queue
from threading import Thread

#from .tasks import ProcessDownload
#from .tasks import query_data

# def evaluation_submit(request): 
#     if request.method == 'POST': 
#         #evaluation_results = json.loads(request.body)  
#         dropdown1_value = request.POST.get('评价1')
#         # for result in evaluation_results: 
#         #     command = result['command'] 
#         #     keyword = result['keyword'] 
#         #     evaluation = result['evaluation']  
#         #return JsonResponse({'message': 'Evaluation results submitted successfully: %s' % request.body.decode('utf-8')}) 
#         return JsonResponse({'message': 'Evaluation results submitted successfully: %s' % dropdown1_value})
#         #return HttpResponse(json.dump(evaluation_results))
#     else: 
#         return JsonResponse({'error': 'Invalid request method.'}, status=400) 

def pyecharts_chart(request):
    days=int(request.GET.get('days'))
    exclude_condition1 = ~Q(user_ip__icontains='10.90.248.214')
    exclude_condition2 = ~Q(user_ip__icontains='10.90.178.13')
    exclude_condition3 = ~Q(user_ip__icontains='10.90.249.119')
    today = datetime.datetime.now().date()
    date_list_15 = [today - datetime.timedelta(days=x) for x in range(days)]
    view_num_dict={}
    user_num_dict={}
    for date in date_list_15:
        active_condition = Q(created_at__date=date)
        combined_condition = active_condition & exclude_condition1 & exclude_condition2 & exclude_condition3
        obj=ConfigSearch.objects.filter(combined_condition)
        view_num=len(obj)
        user_ip_list=[]
        for item in obj:
            if item.user_ip not in user_ip_list:
                user_ip_list.append(item.user_ip)
        user_num=len(user_ip_list)
        view_num_dict[date]=view_num
        user_num_dict[date]=user_num
    date_list=[]
    view_num_list=[]
    user_num_list=[]
    for k,v in view_num_dict.items():
        date_obj=k
        date_list.append(date_obj.strftime('%m-%d'))
        view_num_list.append(v)
        user_num_list.append(user_num_dict[k])
    date_list.reverse()
    view_num_list.reverse()
    user_num_list.reverse()
    view_num_average_list=[]
    for i in range(len(view_num_list)):
        if user_num_list[i]!=0:
            view_num_average_list.append(int(view_num_list[i]/user_num_list[i]))
        else:
            view_num_average_list.append(0)
        
    # x_list = []
    # y_list = []
    # z_list = []
    # for i in range(len(date_list)):
    #     x_list.append(date_list[i])
    #     y_list.append(int(user_num_list[i]))
    #     z_list.append(int(view_num_list[i]))
    bar = (
        Bar(init_opts=opts.InitOpts(width="1200px", height="300px"))
        .add_xaxis(date_list)
        .add_yaxis("访问用户数", user_num_list)
        
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0", is_show=True),
            # title_opts=opts.TitleOpts(
            #     title="页面访问数据统计图",
            # ),
        )
    )
    line = (
        Line(init_opts=opts.InitOpts(width="1200px", height="300px"))
        .add_xaxis(date_list)
        .add_yaxis("用户访问次数", view_num_list)
        
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0", is_show=True),
            # title_opts=opts.TitleOpts(
            #     title="页面访问用户数量15日统计图",
            # ),
        )
    )
    line_average = (
        Line(init_opts=opts.InitOpts(width="1200px", height="300px"))
        .add_xaxis(date_list)
        .add_yaxis("人均访问次数", view_num_average_list)
        
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0", is_show=True),
            # title_opts=opts.TitleOpts(
            #     title="页面访问用户数量15日统计图",
            # ),
        )
    )
    page = Page()
    page.add(bar)
    page.add(line)
    page.add(line_average)
    return HttpResponse(page.render_embed())

def keyword_search_chart(request):
    days=int(request.GET.get('days'))
    exclude_condition1 = ~Q(user_ip__icontains='10.90.248.214')
    exclude_condition2 = ~Q(user_ip__icontains='10.90.178.13')
    exclude_condition3 = ~Q(user_ip__icontains='10.90.249.119')
    today = datetime.datetime.now().date()
    date_list_15 = [today - datetime.timedelta(days=x) for x in range(days)]
    view_num_dict={}
    user_num_dict={}
    for date in date_list_15:
        active_condition = Q(created_at__date=date)
        combined_condition = active_condition & exclude_condition1 & exclude_condition2 & exclude_condition3
        obj=KeywordSearch.objects.filter(combined_condition)
        view_num=len(obj)
        user_ip_list=[]
        for item in obj:
            if item.user_ip not in user_ip_list:
                user_ip_list.append(item.user_ip)
        user_num=len(user_ip_list)
        view_num_dict[date]=view_num
        user_num_dict[date]=user_num
    date_list=[]
    view_num_list=[]
    user_num_list=[]
    for k,v in view_num_dict.items():
        date_obj=k
        date_list.append(date_obj.strftime('%m-%d'))
        view_num_list.append(v)
        user_num_list.append(user_num_dict[k])
    date_list.reverse()
    view_num_list.reverse()
    user_num_list.reverse()
    view_num_average_list=[]
    for i in range(len(view_num_list)):
        if user_num_list[i]!=0:
            view_num_average_list.append(int(view_num_list[i]/user_num_list[i]))
        else:
            view_num_average_list.append(0)
    # x_list = []
    # y_list = []
    # z_list = []
    # for i in range(len(date_list)):
    #     x_list.append(date_list[i])
    #     y_list.append(int(user_num_list[i]))
    #     z_list.append(int(view_num_list[i]))
    bar = (
        Bar(init_opts=opts.InitOpts(width="1200px", height="300px"))
        .add_xaxis(date_list)
        .add_yaxis("访问用户数", user_num_list)
        
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0", is_show=True),
            # title_opts=opts.TitleOpts(
            #     title="页面访问数据统计图",
            # ),
        )
    )
    line = (
        Line(init_opts=opts.InitOpts(width="1200px", height="300px"))
        .add_xaxis(date_list)
        .add_yaxis("用户访问次数", view_num_list)
        
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0", is_show=True),
            # title_opts=opts.TitleOpts(
            #     title="页面访问用户数量15日统计图",
            # ),
        )
    )
    line_average = (
        Line(init_opts=opts.InitOpts(width="1200px", height="300px"))
        .add_xaxis(date_list)
        .add_yaxis("人均访问次数", view_num_average_list)
        
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0", is_show=True),
            # title_opts=opts.TitleOpts(
            #     title="页面访问用户数量15日统计图",
            # ),
        )
    )
    page = Page()
    page.add(bar)
    page.add(line)
    page.add(line_average)
    return HttpResponse(page.render_embed())

def evaluation_submit_json(request): 
    if request.method == 'POST': 
        evaluation_results = json.loads(request.body)
        user_ip = request.META.get('REMOTE_ADDR')
        results_dict={}
        if len(evaluation_results)>=1:
            evaluation_list = evaluation_results[0]['evaluation']
            i=0
            for item in evaluation_results:
                if item['command'].startswith('!<'):
                    results_dict[item['command']]={'evaluation': evaluation_list[i]['evaluation'], 'keyword': item['keyword']}
                    i=i+1
            for k, v in results_dict.items():
                result_feedback_obj = ResultFeedback(command=k, keyword=v['keyword'], evaluation=v['evaluation'], user_ip=user_ip)
                result_feedback_obj.save()
            return JsonResponse({'message': '评价结果已提交。'})
            #return HttpResponse('评价结果已提交。')
            #return JsonResponse({'message': 'Evaluation json results submitted successfully: %s' % results_dict})
        #return HttpResponse('%s' % results_dict)
        else:
            return HttpResponse('提交的评价结果为空。')
    else: 
        return JsonResponse({'error': 'Invalid request method.'}, status=400) 

def demo_view(request):
	# If method is POST, process form data and start task
	if request.method == 'POST':
		# Create Task
		download_task = ProcessDownload.delay()
		# Get ID
		task_id = download_task.task_id
		# Print Task ID
		print(f'Celery Task ID: {task_id}')
		# Return demo view with Task ID
		return render(request, 'progress.html', {'task_id': task_id})
	else:
		# Return demo view
		return render(request, 'progress.html', {})

def check_task_status(request): 
    if request.method == 'POST': 
        task_id = request.POST.get('task_id') # 获取任务状态 
        task = AsyncResult(task_id) 
        return JsonResponse({'status': task.status}) 
    else: 
        return HttpResponseNotAllowed(['POST']) 

def process_item(item, model, temperature): 
    k, v = item 
    v = v[0] 
    if k != None: 
        time.sleep(10)
        sample = get_multi_samples(v, to_json=True) 
        time.sleep(10)
        one_keyword = get_multi_scripts_examples_chat(k, sample, model=model, temperature=temperature) 
        if one_keyword == '缺少示例': 
            one_keyword = v + ' ' + '缺少示例' 
        if one_keyword == '输出错误，不符合json格式': 
            one_keyword = v + ' ' + '输出错误，不符合json格式' 
        return (k, [v, one_keyword]) 
    else: 
        return None

def get_keywords_multiprocessing(text, **kwargs):
    start_time = time.time()
    temperature = kwargs.get('temperature') 
    model = kwargs.get('model') 
    query = text 
    config_list = split_config(query) 
    result = {} 
    if config_list != []: 
        for x in config_list: 
            result[x] = [chat_knowledge(x, 'h10-12-DC')] 
        keywords = [] 
        pool = multiprocessing.Pool() 
        items = result.items() 

        # 使用functools.partial来传递额外的参数
        process_item_with_args = partial(process_item, model=model, temperature=temperature)

        # 使用新的函数来调用pool.map
        results = pool.map(process_item_with_args, items) 
        pool.close() 
        pool.join() 
        for res in results: 
            if res is not None: 
                k, v = res 
                keywords = keywords + v[1].split('\n') 
                result[k].append(v[1]) 
        keywords = [x.lstrip(' \t-') for x in keywords] 
        keywords = "" + "\n ".join(keywords) 
    keywords_list=[] 
    for k, v in result.items(): 
        keywords_list.append(v[0]) 
    end_time = time.time()
    execution_time = round(end_time - start_time, 1)
    return keywords_list, result,execution_time

# 线程装饰器
def thread(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        thread = Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread
    return new_func

@thread
def chat_knowledge_threaded(x, queue):
    """工作线程，调用chat_knowledge并将结果放入队列"""
    result = chat_knowledge(x, 'h12-14-remove-dup')
    queue.put((x, result))

# 使用线程装饰器的函数
@thread
def get_multi_scripts_examples_chat_threaded(k, sample, queue, temperature=None, model=None):
    result = get_multi_scripts_examples_chat(k, sample, temperature=temperature, model=model)
    queue.put((k, result))  # 将键k传递给队列

def get_keywords_multithread(text, **kwargs):
    #start_time = time.time()
    temperature = kwargs.get('temperature')
    model = kwargs.get('model')
    query = text
    config_list = split_config(query)
    result = {}
    if config_list != []:
        threads = []
        queue = Queue()
        for x in config_list:
            #result[x] = [chat_knowledge(x, 'h12-14-remove-dup')]
            thread = chat_knowledge_threaded(x, queue)  # 这里不需要显式创建Thread实例
            threads.append(thread)
        for thread in threads:
            thread.join()

        # 从队列中收集结果
        while not queue.empty():
            x, res = queue.get()
            result[x] = [res]
        keywords = []
        threads = []
        queue = Queue()
        for k,v in result.items():
            v=v[0]
            if k !=None:
                sample = get_multi_samples(v, to_json=True)
                thread = get_multi_scripts_examples_chat_threaded(k, sample, queue, temperature=temperature, model=model)
                threads.append(thread)
        for thread in threads:
            thread.join()
        while not queue.empty():
            k, thread_result = queue.get()  # 获取键k和结果
            if thread_result == '缺少示例':
                thread_result = v + '    ' + '缺少示例'
            if thread_result == '输出错误，不符合json格式':
                thread_result = v + '    ' + '输出错误，不符合json格式'
            keywords = keywords + thread_result.split('\n')
            result[k].append(thread_result)  # 使用键k更新结果
        keywords = [x.lstrip(' \t-') for x in keywords]
        keywords = "" + "\n    ".join(keywords)
    keywords_list=[]
    for k, v in result.items():
        keywords_list.append(v[0])
    #end_time = time.time()
    #execution_time = round(end_time - start_time, 1)
    return keywords_list,result

def chat_knowledge_for_show_command(query, knowledge_base, top_k=1, raw_data=False):
    url = 'http://10.227.153.211:8960/chat/knowledge_base_chat'  # 这里需要改成langchain服务的地址
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    query = query
    knowledge_base = knowledge_base
    data = {
        "knowledge_base_name": f"{knowledge_base}",
        "top_k": top_k,
        "query": f"{query}",
        "history": [
            {
                "role": "user",
                "content": "根据已知信息，输出关键字,只使用已知信息中的词汇"
            },
            {
                "role": "assistant",
                "content": ""
            }
        ],
        "stream": False,
        "local_doc_url": False,
        "score_threshold": 1,
        "model_name": "Qwen_14B_Chat",
        #"prompt_template": "<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n\n<已知信息>{{ context }}</已知信息>\n\n<问题>{{ question }}</问题>"
        "prompt_template": "只回答**知道了**"
    }
    response = requests.post(url, headers=headers,
                             data=json.dumps(data), stream=True)
    print(f"向量匹配参数：{data}")
    if response.status_code == 200:
        content = ''
        start_time = time.time()
        for line in response.iter_content(decode_unicode=True):
            content = content+line
        try:
            # print(f'this is {content}')
            content = json.loads(content.lstrip('data: '))
        except Exception as e:
            print(e)
        end_time = time.time()
        execution_time = round(end_time - start_time, 1)
        print(f"查询的耗时是：{execution_time}")
        if content.get('code', None) == 404:
            raise Exception(content['msg'])
        else:
            if not raw_data:
                value_list = []
                show_command_list=[]
                #print(content['docs'])
                for item in content['docs']:
                    match = re.search(r'关键字名称: (.*)\n', item)
                    #if match.group(1) not in value_list:
                    value_list.append(match.group(1))
                    match = re.search(r'show命令行: (.*)\n', item)
                    #if match.group(1) not in show_command_list:
                    show_command_list.append(match.group(1))
                return (value_list, show_command_list, execution_time)
            else:
                result = content['docs']
                print(result, type(result))
                return (result, execution_time)
    else:
        print("Error:", response.status_code)

def chat_knowledge_for_keyword_doc(query, knowledge_base, top_k=1, raw_data=False):
    url = 'http://10.227.153.211:8960/chat/knowledge_base_chat'  # 这里需要改成langchain服务的地址
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    query = query
    knowledge_base = knowledge_base
    data = {
        "knowledge_base_name": f"{knowledge_base}",
        "top_k": top_k,
        "query": f"{query}",
        "history": [
            {
                "role": "user",
                "content": "根据已知信息，输出关键字,只使用已知信息中的词汇"
            },
            {
                "role": "assistant",
                "content": ""
            }
        ],
        "stream": False,
        "local_doc_url": False,
        "score_threshold": 1,
        "model_name": "Qwen_14B_Chat",
        #"prompt_template": "<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n\n<已知信息>{{ context }}</已知信息>\n\n<问题>{{ question }}</问题>"
        "prompt_template": "只回答**知道了**"
    }
    response = requests.post(url, headers=headers,
                             data=json.dumps(data), stream=True)
    print(f"向量匹配参数：{data}")
    if response.status_code == 200:
        content = ''
        start_time = time.time()
        for line in response.iter_content(decode_unicode=True):
            content = content+line
        try:
            # print(f'this is {content}')
            content = json.loads(content.lstrip('data: '))
        except Exception as e:
            print(e)
        end_time = time.time()
        execution_time = round(end_time - start_time, 1)
        print(f"查询的耗时是：{execution_time}")
        if content.get('code', None) == 404:
            raise Exception(content['msg'])
        else:
            if not raw_data:
                keyword_name_list = []
                #print(content['docs'])
                for item in content['docs']:
                    match = re.search(r'关键字名称: (.*)\n', item)
                    if match:
                        keyword_name_list.append(match.group(1))
                return (keyword_name_list, execution_time)
                #return keyword_name_list
            else:
                result = content['docs']
                print(result, type(result))
                return (result, execution_time)
    else:
        print("Error:", response.status_code)

def chat_knowledge_for_ceshiyi_keyword_testcase(query, knowledge_base, top_k=1, raw_data=False):
    url = 'http://10.227.153.211:8960/chat/knowledge_base_chat'  # 这里需要改成langchain服务的地址
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    query = query
    knowledge_base = knowledge_base
    data = {
        "knowledge_base_name": f"{knowledge_base}",
        "top_k": top_k,
        "query": f"{query}",
        "history": [
            {
                "role": "user",
                "content": "根据已知信息，输出关键字,只使用已知信息中的词汇"
            },
            {
                "role": "assistant",
                "content": ""
            }
        ],
        "stream": False,
        "local_doc_url": False,
        "score_threshold": 1,
        "model_name": "Qwen_14B_Chat",
        #"prompt_template": "<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n\n<已知信息>{{ context }}</已知信息>\n\n<问题>{{ question }}</问题>"
        "prompt_template": "只回答**知道了**"
    }
    response = requests.post(url, headers=headers,
                             data=json.dumps(data), stream=True)
    print(f"向量匹配参数：{data}")
    if response.status_code == 200:
        content = ''
        start_time = time.time()
        for line in response.iter_content(decode_unicode=True):
            content = content+line
        try:
            # print(f'this is {content}')
            content = json.loads(content.lstrip('data: '))
        except Exception as e:
            print(e)
        end_time = time.time()
        execution_time = round(end_time - start_time, 1)
        print(f"查询的耗时是：{execution_time}")
        if content.get('code', None) == 404:
            raise Exception(content['msg'])
        else:
            if not raw_data:
                ceshiyi_testcase_list = []
                #print(content['docs'])
                for item in content['docs']:
                    search_result = re.search(r'测试仪关键字示例:(.*?)测试仪关键字列表:', item, re.DOTALL)
                    if search_result:
                        ceshiyi_testcase_content = '    ' + search_result.group(1).strip()
                        ceshiyi_testcase_list.append(ceshiyi_testcase_content)
                return (ceshiyi_testcase_list, execution_time)
            else:
                result = content['docs']
                print(result, type(result))
                return (result, execution_time)
    else:
        print("Error:", response.status_code)

def get_keywords_new(text, **kwargs):
    temperature = kwargs.get('temperature')
    model = kwargs.get('model')
    query = text
    config_list = split_config(query)
    result = {}
    if config_list != []:
        for x in config_list:
            if model=='Qwen':
                result[x] = ['']
            else:
                result[x] = [chat_knowledge(x, 'h12-14-remove-dup')]
        keywords = []
        for k,v in result.items():
            v=v[0]
            if k !=None:
                sample = get_multi_samples(v, to_json=True)
                if model=='Qwen':
                    one_keyword = get_keyword_chat(k, temperature=temperature, model=model)
                else:
                    one_keyword = get_multi_scripts_examples_chat(k, sample, temperature=temperature, model=model)
                if one_keyword == '缺少示例':
                    one_keyword = v + '    ' + '缺少示例'
                if one_keyword == '输出错误，不符合json格式':
                    one_keyword = v + '    ' + '输出错误，不符合json格式'
                keywords = keywords + one_keyword.split('\n')
                if model=='Qwen':
                    result[k][0]=one_keyword.split('    ')[0]
                result[k].append(one_keyword)
                
            else:
                continue
        keywords = [x.lstrip(' \t-') for x in keywords]
        keywords = "" + "\n    ".join(keywords)
    keywords_list=[]
    for k, v in result.items():
        if model=='Qwen':
            keywords_list.append(v[1].split('    ')[0])
        else:
            keywords_list.append(v[0])
    return keywords_list,result

def countdown(request):
    now = datetime.datetime.now()
    # 设置倒计时结束时间（10分钟）
    end_time = now + datetime.timedelta(minutes=1)
    context = {'end_time': end_time}
    return render(request, 'countdown.html', context)

@api_view(['GET']) 
def check_keyword_name(request): 
    keyword_name=request.GET.get('keyword_name')
    try: 
        obj = KeywordDoc.objects.get(keyword_name=keyword_name) 
        return Response(True) 
    except KeywordDoc.DoesNotExist: 
        return Response(False)

def process_func(x): 
    return x, chat_knowledge(x.replace('<','[').replace('>',']'), 'h10-08+DC')

def split_config(configtext):
    configtrans = ConfigTrans(configtext)
    config_list = configtrans._export_config_split_list()
    static_route_command_list=[]
    static_command_list=[]
    for item in config_list:
        if item.startswith('!<static>'):
            config_list.remove(item)
            temp_list=item.split('\n')
            temp_list_clone=item.split('\n')
            for temp_item in temp_list:
                if temp_item.startswith('ip route ') or temp_item.startswith('ip route-static ') or temp_item.startswith('ip route-vxlan '):
                    temp_list_clone.remove(temp_item)
                    static_route_command_list.append(temp_item)
            if len(temp_list_clone)>2:
                static_command_list.append(temp_list)
    for static_route_command in static_route_command_list:
        config_list.append('!<static>\n%s\n!</static>' % static_route_command)
    if len(static_command_list)!=0:
        for static_command in static_command_list:
            config_list=config_list+static_command
    return config_list

def get_keyword(content):
    text = content
    keyword = "keyword:"
    start_index = text.find(keyword)
    if start_index != -1:
        extracted_info = text[start_index + len(keyword):].strip()
        return f'{extracted_info}'
    else:
        return f'{""}'

def chat_knowledge(query, knowledge_base):
    url = 'http://10.227.153.211:8960/chat/knowledge_base_chat'  # 这里需要改成langchain服务的地址
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    query = query
    knowledge_base = knowledge_base
    data = {
        "query": f"{query}",
        "knowledge_base_name": f"{knowledge_base}",
        "top_k": 2,
        "history": [
            {
            "role": "user",
            "content": "根据已知信息，输出关键字,只使用已知信息中的词汇"
            },
            {
            "role": "assistant",
            "content": "配置interface"
            }
        ],
        "stream": False,
        "local_doc_url": False,
        "score_threshold": 1,
        "model_name": "Baichuan2-13B-Chat",
        #"prompt_template": "<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n\n<已知信息>{{ context }}</已知信息>\n\n<问题>{{ question }}</问题>"
        "prompt_template": "只回答**知道了**"
        }
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    if response.status_code == 200:
        content = ''
        for line in response.iter_content(decode_unicode=True):
            content = content+line
        try:
            # print(f'this is {content}')
            content = json.loads(content.lstrip('data: '))
        except Exception as e:
            print(e)
            
        if content.get('code',None) == 404:
            raise Exception(content['msg'])
        else:
            key = get_keyword(content['docs'][0])
            return key
    else:
        print("Error:", response.status_code)
        

def get_keywords(text, **kwargs):
    query = text
    config_list = split_config(query)
    result = {}
    if config_list != []:
        # for x in config_list:
        #     result[x] = chat_knowledge(x.replace('<','[').replace('>',']'), 'h9-15')
        total_tasks = len(config_list) 
        completed_tasks = multiprocessing.Value('i', 0) 
        pool = multiprocessing.Pool() 
        results = pool.map(process_func, config_list) 
        result = {x: res for x, res in results}
        with completed_tasks.get_lock(): 
            completed_tasks.value += 1 
            progress = completed_tasks.value / total_tasks 
            print(f"当前进度：{progress * 100}%")
            #return redirect('countdown')
    keywords=[]
    for k, v in result.items():
        keywords.append(v)
    return keywords,result

# def get_keyword(content):
#     text = content
#     keyword = "keyword:"

#     # 找到关键词的位置
#     start_index = text.find(keyword)

#     if start_index != -1:
#         # 提取关键词后面的信息
#         extracted_info = text[start_index + len(keyword):].strip()
#         # print("提取的信息:", extracted_info)

#         return f'{extracted_info}'
#     else:
#         return f'{""}'

# def chat_knowledge(query, knowledge_base):
#     url = 'http://10.227.153.211:8946/chat/knowledge_base_chat'  # 这里需要改成langchain服务的地址
#     headers = {
#         'accept': 'application/json',
#         'Content-Type': 'application/json',
#     }
#     query = query
#     knowledge_base = knowledge_base
#     data = {
#         "query": f"{query}",
#         "knowledge_base_name": f"{knowledge_base}",
#         "top_k": 5,
#         "history": [
#             {
#             "role": "user",
#             "content": "根据已知信息，输出关键字,只使用已知信息中的词汇"
#             },
#             {
#             "role": "assistant",
#             "content": "配置interface"
#             }
#         ],
#         "stream": False,
#         "local_doc_url": False,
#         "score_threshold": 1,
#         "model_name": "Baichuan2-13B-Chat",
#         "prompt_template": "<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n\n<已知信息>{{ context }}</已知信息>\n\n<问题>{{ question }}</问题>"
#         }
#     response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
#     return response



def generate_random_dict(original_dict, num_elements): 
    keys = list(original_dict.keys()) 
    random.shuffle(keys) 
    new_dict = {} 
    for i in range(num_elements): 
        key = keys[i] 
        new_dict[key] = original_dict[key] 
    return new_dict

def get_sorted_search_keys():
    target_date = datetime.datetime.now().date()
    #search_counter_today = len(KeywordSearch.objects.filter(created_at__date=target_date)) + len(ConfigSearch.objects.filter(created_at__date=target_date))
    search_counter_today = len(ConfigSearch.objects.filter(created_at__date=target_date))
    results = KeywordSearch.objects.all()
    results_config = ConfigSearch.objects.all()
    #search_counter_all = len(results) + len(results_config)
    search_counter_all = len(results_config)
    keyword_list = []
    config_list = []
    user_ip_list = []
    # for item in results:
    #     if item.keyword_search_item!='':
    #         keyword_list.append(item.keyword_search_item)
    #     if item.user_ip!='':
    #         if item.user_ip not in user_ip_list:
    #             user_ip_list.append(item.user_ip)
    # counter = Counter(keyword_list)
    # keyword_dict = dict(counter)
    # sorted_items = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
    # sorted_keys = [item[0].replace('\t', ' ') if '\t' in item[0] else item[0] for item in sorted_items]
    # user_ip_counter = len(user_ip_list)
    # return sorted_keys, search_counter_today, search_counter_all, user_ip_counter

    pattern = r'^命令行配置\[.*\]'
    for item in results:
        if item.keyword_search_item!='':
            if not re.search(pattern, item.keyword_search_item):
                keyword_list.append(item.keyword_search_item)
            # else:
            #     config_list.append(item.keyword_search_item)
        if item.user_ip!='':
            if item.user_ip not in user_ip_list:
                user_ip_list.append(item.user_ip)
    for item_config in results_config:
        if item_config.config_search_item!='':
            config_list.append(item_config.config_search_item.rstrip('\r\n'))
        if item_config.user_ip!='':
            if item_config.user_ip not in user_ip_list:
                user_ip_list.append(item_config.user_ip)
    counter = Counter(keyword_list)
    keyword_dict = dict(counter)
    sorted_items = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_keys = [item[0].replace('\t', ' ') if '\t' in item[0] else item[0] for item in sorted_items]
    counter = Counter(config_list)
    config_dict = dict(counter)
    sorted_config_items = sorted(config_dict.items(), key=lambda x: x[1], reverse=True)
    #sorted_config_keys = [item[0].replace('命令行配置', '') if '命令行配置' in item[0] else item[0] for item in sorted_config_items]
    sorted_config_keys = [item[0].replace(' ', '&nbsp;') if ' ' in item[0] else item[0] for item in sorted_config_items]
    sorted_config_keys = [item.replace('<', '&lt;') if '<' in item else item for item in sorted_config_keys]
    sorted_config_keys = [item.replace('>', '&gt;') if '>' in item else item for item in sorted_config_keys]
    sorted_config_keys = [item.replace('命令行配置[', '') if '命令行配置[' in item else item for item in sorted_config_keys]
    sorted_config_keys = [item.replace(']', '') if ']' in item else item for item in sorted_config_keys]
    sorted_config_keys = [item.replace('\n', '<br>') if '\n' in item else item for item in sorted_config_keys]
    user_ip_counter = len(user_ip_list)
    return sorted_keys, search_counter_today, search_counter_all, user_ip_counter, sorted_config_keys
    


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    return ip

def index_keywordinfo(request):
    context = {}
    context["ip"] = get_ip_address()
    return render(request, "keywordinfo.html", context)

@shared_task(bind=True)
#@app.task
def query_data(self,keyword,keyword_search_range,ai_enable,model,temperature):
    start_time = time.time()
    keyword_list = []
    config_dict = {}
    config_dict_convert = {}
    if ai_enable=='ai-enable':
        #keyword_list,config_dict = get_keywords_new(keyword, model='Qwen_14B_Chat', temperature=0.5)
        keyword_list,config_dict = get_keywords_multithread(keyword, model=model, temperature=temperature)
        #keyword_list,config_dict = get_keywords_new(keyword, model=model, temperature=temperature)
        #keyword_list,config_dict = get_keywords(keyword)
        keyword_list = ["加" + item if item.startswith("配置") else item for item in keyword_list]
        #config_dict_convert = {k: "加" + v if v.startswith("配置") else v for k, v in config_dict.items()}
        config_dict_convert = {k: [f"加{v}" if v.startswith("配置") else v for v in value] for k, value in config_dict.items()}

    elif ai_enable=='ai-disable':
        if '\n' in keyword:
            keyword_list = []
            config_dict_convert = {}
        else:
            query_list = []
            if keyword_search_range=='yes':
                if '|' in keyword: 
                    query_list = ["(Q(keyword_name__icontains='%s')|Q(keyword_doc_html__icontains='%s'))" % (element.strip(),element.strip()) for element in keyword.split('|')]
                    q_item = '|'.join(query_list)
                elif '&' in keyword:
                    query_list = ["(Q(keyword_name__icontains='%s')|Q(keyword_doc_html__icontains='%s'))" % (element.strip(),element.strip()) for element in keyword.split('&')]
                    q_item = '&'.join(query_list)
                else:
                    q_item = "Q(keyword_name__icontains='%s')|Q(keyword_doc_html__icontains='%s')" % (keyword.strip(),keyword.strip())
            elif keyword_search_range=='no':
                if '|' in keyword: 
                    query_list = ["Q(keyword_name__icontains='%s')" % element.strip() for element in keyword.split('|')]
                    q_item = '|'.join(query_list)
                elif '&' in keyword:
                    query_list = ["Q(keyword_name__icontains='%s')" % element.strip() for element in keyword.split('&')]
                    q_item = '&'.join(query_list)
                else:
                    q_item = "Q(keyword_name__icontains='%s')" % (keyword.strip())
            q_object = eval(q_item)
            results = (
                KeywordDoc.objects.filter(q_object)
                .distinct()
                .order_by("keyword_name")
                )
            for item in results:
                keyword_list.append(item.keyword_name)
    end_time = time.time()
    run_time = round(end_time - start_time, 1)
    return keyword_list,config_dict_convert,run_time

def query_data_obj(keyword,keyword_search_range,ai_enable):
    keyword_list = []
    query_list = []
    if ai_enable=='ai-disable':
        if keyword_search_range=='yes':
            if '|' in keyword: 
                query_list = ["(Q(keyword_name__icontains='%s')|Q(keyword_doc_html__icontains='%s'))" % (element.strip(),element.strip()) for element in keyword.split('|')]
                q_item = '|'.join(query_list)
            elif '&' in keyword:
                query_list = ["(Q(keyword_name__icontains='%s')|Q(keyword_doc_html__icontains='%s'))" % (element.strip(),element.strip()) for element in keyword.split('&')]
                q_item = '&'.join(query_list)
            else:
                q_item = "Q(keyword_name__icontains='%s')|Q(keyword_doc_html__icontains='%s')" % (keyword.strip(),keyword.strip())
        elif keyword_search_range=='no':
            if '|' in keyword: 
                query_list = ["Q(keyword_name__icontains='%s')" % element.strip() for element in keyword.split('|')]
                q_item = '|'.join(query_list)
            elif '&' in keyword:
                query_list = ["Q(keyword_name__icontains='%s')" % element.strip() for element in keyword.split('&')]
                q_item = '&'.join(query_list)
            else:
                q_item = "Q(keyword_name__icontains='%s')" % (keyword.strip())
        q_object = eval(q_item)
        results = (
            KeywordDoc.objects.filter(q_object)
            .distinct()
            .order_by("keyword_name")
            )
    else:
        results=()
    return results

#@cache_page(60 * 15)
def search_view(request):
    if request.method == 'GET':
        keyword = request.GET.get('keyword', None)
        db_name = request.GET.getlist('db_name[]')
        keyword_type = request.GET.get('keyword_type')
        keyword_search_range = request.GET.get('keyword_search_range')
        ai_enable = request.GET.get('ai_enable')
        # cache_key = f'search_view_{keyword}_{db_name}_{keyword_type}_{keyword_search_range}'
        # cached_data = cache.get(cache_key)
        # if cached_data:
        #     matches=cached_data
        #     return JsonResponse(matches, safe=False)
        # else:
        matches=[]
        keyword_dict={}
        results = query_data_obj(keyword,keyword_search_range,ai_enable)
        if results:
            for result in results:
                keyword_dict['%s' % result.keyword_name]=int(result.keyword_example_num)
            sorted_keyword_dict = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
            matches = ['关键字速查列表（搜索到%s个RF关键字）:' % len(sorted_keyword_dict)] + [item[0] for item in sorted_keyword_dict]
        else:
            matches=['没有匹配的关键字']
        #cache.set(cache_key, matches)
        return JsonResponse(matches, safe=False)
            

def keywordinfo(request):  # ajax的url
    data_list = []
    peizhi_keyword_dict = {}
    jiancha_keyword_dict = {}
    baowenjiexi_keyword_dict = {}
    ceshiyi_keyword_dict = {}
    for keyword_item in KeywordList.objects.all():
        keyword_item_lower=keyword_item.keyword_name.lower()
        keyword_type = '配置'
        if (re.match('^加%s' % keyword_type, keyword_item_lower)) or (re.match('^删%s' % keyword_type , keyword_item_lower)) or (re.match('^%s' % keyword_type , keyword_item_lower)):
            if keyword_item.keyword_name not in peizhi_keyword_dict:
                peizhi_keyword_dict['%s' % keyword_item.keyword_name]=int(keyword_item.keyword_example_num)
        keyword_type = '检查'
        if keyword_item.keyword_loc:
            if (re.search('/%s' % keyword_type, keyword_item.keyword_loc)):
                if keyword_item.keyword_name not in jiancha_keyword_dict:
                    jiancha_keyword_dict['%s' % keyword_item.keyword_name]=int(keyword_item.keyword_example_num)
        keyword_type = '报文解析'
        if keyword_item.keyword_loc:
            if (re.search('/%s' % keyword_type, keyword_item.keyword_loc)) or (re.search('/CheckPacket', keyword_item.keyword_loc)):
                if keyword_item.keyword_name not in baowenjiexi_keyword_dict:
                    baowenjiexi_keyword_dict['%s' % keyword_item.keyword_name]=int(keyword_item.keyword_example_num)
        keyword_type = '测试仪'
        if re.match('^%s' % keyword_type , keyword_item_lower):
            if keyword_item.keyword_name not in ceshiyi_keyword_dict:
                ceshiyi_keyword_dict['%s' % keyword_item.keyword_name]=int(keyword_item.keyword_example_num)   
                 
    lengths=[len(d) for d in (peizhi_keyword_dict,jiancha_keyword_dict,baowenjiexi_keyword_dict,ceshiyi_keyword_dict)]
    #min_length=min(lengths)
    min_length=100
    id_list = list(range(1,101))
    sorted_peizhi_keyword_list = sorted(peizhi_keyword_dict.items(), key=lambda x: x[1], reverse=True)[:min_length]
    sorted_jiancha_keyword_list = sorted(jiancha_keyword_dict.items(), key=lambda x: x[1], reverse=True)[:min_length]
    sorted_baowenjiexi_keyword_list = sorted(baowenjiexi_keyword_dict.items(), key=lambda x: x[1], reverse=True)[:min_length]
    sorted_ceshiyi_keyword_list = sorted(ceshiyi_keyword_dict.items(), key=lambda x: x[1], reverse=True)[:min_length]
    
    for i in range(min_length):
            data_list.append(
                {
                    "id" : "%s" % id_list[i],
                    "peizhi": "%s" % sorted_peizhi_keyword_list[i][0],
                    "jiancha":"%s" %  sorted_jiancha_keyword_list[i][0],
                    "baowenjiexi": "%s" % sorted_baowenjiexi_keyword_list[i][0],
                    "ceshiyi": "%s" % sorted_ceshiyi_keyword_list[i][0],         
                }
            )
    data_dic = {}
    data_dic["rows"] = data_list  # 格式一定要符合官网的json格式，否则会出现一系列错误
    data_dic["total"] = str(len(data_list))
    data_dic["totalNotFiltered"] = str(len(data_list))
    return HttpResponse(json.dumps(data_dic))

def keywordinfo_all(request):  # ajax的url
    # data_list = []
    # peizhi_keyword_dict = {}
    # jiancha_keyword_dict = {}
    # baowenjiexi_keyword_dict = {}
    # ceshiyi_keyword_dict = {}
    # response_cache_key = f'keywordinfo_all'
    # for keyword_item in KeywordList.objects.all():
        
    #     if int(keyword_item.keyword_example_num)!=0:
    #         keyword_item_lower=keyword_item.keyword_name.lower()
    #         keyword_type = '配置'
    #         if (re.match('^加%s' % keyword_type, keyword_item_lower)) or (re.match('^删%s' % keyword_type , keyword_item_lower)) or (re.match('^%s' % keyword_type , keyword_item_lower)):
    #             if keyword_item.keyword_name not in peizhi_keyword_dict:
    #                 peizhi_keyword_dict['%s' % keyword_item.keyword_name]=int(keyword_item.keyword_example_num)
    #         keyword_type = '检查'
    #         if keyword_item.keyword_loc:
    #             if (re.search('/%s' % keyword_type, keyword_item.keyword_loc)):
    #                 if keyword_item.keyword_name not in jiancha_keyword_dict:
    #                     jiancha_keyword_dict['%s' % keyword_item.keyword_name]=int(keyword_item.keyword_example_num)
    #         keyword_type = '报文解析'
    #         if keyword_item.keyword_loc:
    #             if (re.search('/%s' % keyword_type, keyword_item.keyword_loc)) or (re.search('/CheckPacket', keyword_item.keyword_loc)):
    #                 if keyword_item.keyword_name not in baowenjiexi_keyword_dict:
    #                     baowenjiexi_keyword_dict['%s' % keyword_item.keyword_name]=int(keyword_item.keyword_example_num)
    #         keyword_type = '测试仪'
    #         if re.match('^%s' % keyword_type , keyword_item_lower):
    #             if keyword_item.keyword_name not in ceshiyi_keyword_dict:
    #                 ceshiyi_keyword_dict['%s' % keyword_item.keyword_name]=int(keyword_item.keyword_example_num)   
                 
    # lengths=[len(d) for d in (peizhi_keyword_dict,jiancha_keyword_dict,baowenjiexi_keyword_dict,ceshiyi_keyword_dict)]
    # max_length=max(lengths)
    # #min_length=100
    # id_list = list(range(1,max_length+1))
    
    # sorted_peizhi_keyword_list = sorted(peizhi_keyword_dict.items(), key=lambda x: x[1], reverse=True)
    # sorted_jiancha_keyword_list = sorted(jiancha_keyword_dict.items(), key=lambda x: x[1], reverse=True)
    # sorted_baowenjiexi_keyword_list = sorted(baowenjiexi_keyword_dict.items(), key=lambda x: x[1], reverse=True)
    # sorted_ceshiyi_keyword_list = sorted(ceshiyi_keyword_dict.items(), key=lambda x: x[1], reverse=True)

    # result = [[elem if elem is not None else '' for elem in arr] for arr in zip_longest(sorted_peizhi_keyword_list, sorted_jiancha_keyword_list, sorted_baowenjiexi_keyword_list, sorted_ceshiyi_keyword_list, fillvalue=['',''])]
    # for i in range(max_length):
    #         data_list.append(
    #             {
    #                 "id" : "%s" % id_list[i],
    #                 "peizhi": "%s" % result[i][0][0],
    #                 "jiancha":"%s" %  result[i][1][0],
    #                 "baowenjiexi": "%s" % result[i][2][0],
    #                 "ceshiyi": "%s" % result[i][3][0],         
    #             }
    #         )
    data_dic = {}
    # data_dic["rows"] = data_list  # 格式一定要符合官网的json格式，否则会出现一系列错误
    # data_dic["total"] = str(len(data_list))
    # data_dic["totalNotFiltered"] = str(len(data_list))
    #cache.set('response_cache_key', json.dumps(data_dic)) 
    # with open("/var/www/LRM/srm/templates/data_dic.json", "w") as file: 
    #     json.dump(data_dic, file) 

    with open("/var/www/LRM/srm/templates/data_dic_new.json", "r") as file:
        data_dic = json.load(file) 
        
    return HttpResponse(json.dumps(data_dic))

def detail_view(request): 
    script_id=int(request.GET.get('id'))
    keyword_name=request.GET.get('keyword_name')
    #obj = get_object_or_404(ScriptPost, id=id) 
    obj = ScriptPost.objects.get(id=script_id) 
    pageview_num = int(obj.pageview)
    
    # content = obj.script
    # content = content.replace('\t', '    ') 
    # html_content = highlight(content, RobotFrameworkLexer(), HtmlFormatter(full=True))
    html_content = '测试用例robot文件路径：' + obj.script_location + obj.script_html
    html_content = html_content.replace(keyword_name, f"<span class='highlight_keyword'>{keyword_name}</span>") 

    html_style='pre { line-height: 125%; }'
    html_style_new='.highlight_keyword { background-color: yellow; }\npre { line-height: 125%; }'
    
    html_content = html_content.replace(html_style, f"{html_style_new}") 
    
    with open('/var/www/LRM/srm/templates/script_%s.html' % script_id, 'w') as file:
        file.write(html_content)
    #script = obj.script.split('\n')
    #context = { 'object': obj , 'content': script} 
    #return render(request, 'detail.html', context) 
    #ScriptPost.objects.update(pageview=0)
    pageview_num = pageview_num + 1
    obj.pageview = pageview_num
    obj.save()
    return render(request, 'script_%s.html' % script_id) 

def testcase_view(request): 
    testcase_id=int(request.GET.get('id'))
    obj = ScriptList.objects.get(id=testcase_id) 
    pageview_num = int(obj.pageview)
    #content = '测试用例名称：%s' % obj.testcase_name + '\n\n' + '测试用例文件路径：%s' % obj.testcase_loc + '\n\n'+ '测试用例文档：' + '\n\n' + obj.testcase_doc + '\n\n' + '测试用例内容：' + '\n\n' + obj.testcase
    #content = content.replace('\t', '    ') 
    #html_content = highlight(content, RobotFrameworkLexer(), HtmlFormatter(full=True))
    html_content = obj.testcase_html
    with open('/var/www/LRM/srm/templates/testcase_%s.html' % testcase_id, 'w') as file:
        file.write(html_content)
    pageview_num = pageview_num + 1
    obj.pageview = pageview_num
    obj.save()
    return render(request, 'testcase_%s.html' % testcase_id) 

def keyword_doc_view(request): 
    keyword_name=request.GET.get('keyword')
    obj = KeywordList.objects.filter(keyword_name='%s' % keyword_name) 
    #doc = obj.keyword_doc_html.split('\n')
    if obj[0].keyword_example:
        example_dict = eval(obj[0].keyword_example)
        id_list=[]
        example_dict_no_dup={}
        index=0
        for key,value in example_dict.items():
            if value[1] not in id_list:
                index=index+1
                example_dict_no_dup[index]=value
                id_list.append(value[1])
        if len(example_dict_no_dup)>100:
            example_dict_no_dup_100 = generate_random_dict(example_dict_no_dup,100)
        else: 
            example_dict_no_dup_100 = example_dict_no_dup     
            
        # for key,value in example_dict.items():
        #     html_content = highlight('\n'.join(value[2]), RobotFrameworkLexer(), HtmlFormatter(full=True))
        #     value[2] = html_content.split('\n')[98:109]
        
    # if obj.testcase_example:
    #     example_dict = eval(obj.testcase_example)
    else:
        example_dict_no_dup = {}
        example_dict_no_dup_100 = {}
    #context = { 'object': obj , 'content': doc, 'content_example' : example_dict, 'keyword': keyword_name} 
    ip = get_ip_address()
    #ip = '10.57.170.40'
    context = { 'object': obj[0], 'content_example' : example_dict_no_dup_100, 'keyword': keyword_name, 'total_number': len(example_dict_no_dup), 'ip': ip} 
    #context = {'keyword': keyword_name} 
    return render(request, 'doc.html', context) 

def case_all_view(request): 
    keyword_name=request.GET.get('keyword')
    obj = KeywordList.objects.filter(keyword_name='%s' % keyword_name) 
    if obj[0].testcase_example:
        example_dict = eval(obj[0].testcase_example)
        id_list=[]
        example_dict_no_dup={}
        index=1
        for key,value in example_dict.items():
            if value[1] not in id_list:
                example_dict_no_dup[index]=value
                id_list.append(value[1])
                index=index+1
        if len(example_dict_no_dup)>100:
            example_dict_no_dup_100 = generate_random_dict(example_dict_no_dup,100)  
        else: 
            example_dict_no_dup_100 = example_dict_no_dup
            
        # for key,value in example_dict.items():
        #     html_content = highlight('\n'.join(value[2]), RobotFrameworkLexer(), HtmlFormatter(full=True))
        #     value[2] = html_content.split('\n')[98:109]
        
    # if obj.testcase_example:
    #     example_dict = eval(obj.testcase_example)
    else:
        example_dict_no_dup = {}
        example_dict_no_dup_100 = {}
    #context = { 'object': obj , 'content': doc, 'content_example' : example_dict, 'keyword': keyword_name} 
    ip = get_ip_address()
    #ip = '10.57.170.40'
    context = { 'object': obj[0], 'content_example' : example_dict_no_dup_100, 'keyword': keyword_name, 'total_number': len(example_dict_no_dup), 'ip': ip} 
    #context = {'keyword': keyword_name} 
    return render(request, 'doc_testcase.html', context) 

def task_result_view(request): 
    task_id=request.GET.get('task_id')
    ip = get_ip_address()
    keyword_type = 'all'
    db_name_list=['keywords','lib']
    keyword_list=[]
    result = AsyncResult(task_id)
    # response_data = {
    #     'progress': result.info.get('progress', 0)
    # }
    # return JsonResponse(response_data)
    #time.sleep(2)
    if result:
        if result.status=='SUCCESS':
            keyword_list_query=result.result[0]
            config_dict_query=result.result[1]
            run_time=result.result[2]
            keyword=''
            query_list = []
            if '|' in keyword: 
                query_list = [keyword.strip() for keyword in keyword.split('|')]
            elif '&' in keyword:
                query_list = [keyword.strip() for keyword in keyword.split('&')]
            else:
                query_list = [keyword.strip()]
            for keyword_item in keyword_list_query:
                keyword_item_lower=keyword_item.lower()
                if keyword_type == 'all':
                    keyword_list.append(keyword_item)
                elif keyword_type == '配置':
                    if (re.match('^加%s' % keyword_type , keyword_item_lower)) or (re.match('^删%s' % keyword_type , keyword_item_lower)) or (re.match('^%s' % keyword_type , keyword_item_lower)):
                        keyword_list.append(keyword_item)
                elif keyword_type == '检查':
                    if (re.match('^%s' % keyword_type , keyword_item_lower)) or (re.match('^获取' , keyword_item_lower)) or (re.match('^重复%s' % keyword_type, keyword_item_lower)):
                        keyword_list.append(keyword_item)
                else:
                    if (re.match('^%s' % keyword_type , keyword_item_lower)):
                        keyword_list.append(keyword_item)
            keyword_list_sorted = sorted(keyword_list, key=len, reverse=True)
            keyword_doc_dict = {}
            results=[]
            for keyword_name in keyword_list_sorted:
                lines=''
                rows=[]
                queryset = KeywordDoc.objects.filter(keyword_name='%s' % keyword_name)
                if queryset:
                    if queryset[0].keyword_loc:
                        if queryset[0].keyword_loc.split('/')[3].lower() in db_name_list:
                            queryset = KeywordDoc.objects.filter(keyword_name='%s' % keyword_name)
                            if queryset[0].keyword_doc_html:
                                highlighted_lines=queryset[0].keyword_doc_html
                                # for query_item in query_list: 
                                #     pattern = query_item
                                #     match = re.search(pattern, highlighted_lines, flags=re.IGNORECASE) 
                                #     if match: 
                                #         matched_string = match.group()
                                #         highlighted_lines = highlighted_lines.replace(matched_string, f"<span class='highlight'>{matched_string}</span>") 
                                #     else:
                                #         highlighted_lines = highlighted_lines.replace(query_item, f"<span class='highlight'>{query_item}</span>")
                                
                                lines = highlighted_lines.split('\n')
                            
                            for query_item in query_list:
                                
                                pattern = query_item
                                match = re.search(pattern, keyword_name, flags=re.IGNORECASE) 
                                if match: 
                                    matched_string = match.group()
                                    keyword_name_highlight = keyword_name.replace(matched_string, f"<span class='highlight'>{matched_string}</span>") 
                                    
                                else:
                                    keyword_name_highlight = keyword_name.replace(query_item, f"<span class='highlight'>{query_item}</span>")
                            script_results_all=[]
                            rows.append(keyword_name_highlight)
                            rows.append(lines)
                            rows.append(keyword_name)
                            keyword_doc_dict['%s' % keyword_name]=[keyword_name_highlight, lines]
                            results.append(rows)
                else:
                    rows.append('')
                    rows.append('')
                    rows.append(keyword_name)
                    keyword_doc_dict['%s' % keyword_name]=['', '']
                    results.append(rows)
            results_invert = []
            for k,v in config_dict_query.items():
                config_rows = []
                config_rows.append(keyword_doc_dict[v[0]][0])
                config_rows.append(keyword_doc_dict[v[0]][1])
                _autolearn = AutoLearn()
                configtext = _autolearn.dryrun_keyword_text(text=v[1])
                configtext = json.loads(configtext)
                
                config_rows.append(v[0])
                k=k.replace('<', '&lt;')
                k=k.replace('>', '&gt;')
                k=k.replace(' ', '&nbsp;')
                k=k.replace('\n', '<br>')
                config_rows.append(k)
                v[1]=v[1].replace(' ', '&nbsp;')
                config_rows.append(v[1])
                execute_code_result = configtext[0]['result']
                execute_code_result=execute_code_result.replace(' ', '&nbsp;')
                execute_code_result=execute_code_result.replace('\n', '<br>')
                if configtext[0]['status']==1:
                    config_rows.append(execute_code_result)
                else:
                    config_rows.append('')
                results_invert.append(config_rows)
            total_number = len(keyword_list_query)
            return render(request, 'search_results_with_config_and_task_id.html', {'results': results_invert, 'total_number': total_number, 'ip': ip, 'run_time': run_time})
        else:
            # return HttpResponse("后台AI推荐程序还未运行完毕，目前处于%s状态，请稍后再点击链接http://%s:7070/task_result?task_id=%s查询推荐结果。" % (result.status, ip, task_id))
            #return render(request, 'search_results_link_with_task_id_refresh.html', {'result': result, 'ip': ip, 'task_id': task_id})
            return render(request, 'search_results_link_with_task_id_refresh.html', {'ip': ip, 'task_id': task_id})
        
    else:
        return render(request, 'search_no_results_link_with_task_id_refresh.html', {'ip': ip, 'task_id': task_id})
    
def replace_quotes(string):
    pattern = r'``([^``]*)``'
    replacement = r'<code>\1</code>'
    result = re.sub(pattern, replacement, string)
    return result

def convert_to_html(text): 
    html = "<p><b>参数介绍：</b></p>\n" 
    if '配置参数介绍' in text:
        option_params = text[text.index("*参数介绍：*") + 7:text.index("*配置参数介绍：*")].strip().split("\n")
    else:
        option_params = text[text.index("*参数介绍：*") + 7:text.index("*作者：*")].strip().split("\n")
        
    option_params_html=replace_quotes(option_params[0])
    option_params_list=option_params_html.split('...')
    for option in option_params_list:
            if '<code>' in option:
                html += "<p>" + option.split('</code>')[0] + "</code>  " + option.split('</code>')[1] + "</p>\n"
            
    html += "<p><b>配置参数介绍：</b></p>\n" 
    html += "<table>\n" 
    html += "<tr><td><b>参数名称</b></td><td><b>参数说明</b></td><td><b>value</b></td><td><b>对应配置或命令</b></td></tr>\n" 
    config_params = text[text.index("*配置参数介绍：*") + 9:text.index("*示例：*")].strip().split("\n") 
    for param in config_params: 
        param = param.strip("|").strip() 
        #print(param)
        if param!="...":
            if "参数名称" not in param:
                html += "<tr>" 
                for i in range(len(param.split("|"))):
                    if param.split("|")[i].strip()!="...":
                        string = replace_quotes(param.split("|")[i].strip())
                        html += "<td>" + string + "</td>" 
            
                html += "</tr>\n" 
    html += "</table>\n" 
    html += "<p><b>示例：</b></p>\n" 
    html += "<table>\n" 
    examples = text[text.index("*示例：*") + 5:text.index("*作者：*")].strip().split("\n") 
    for example in examples: 
        print(example)
        example = example.strip("|").strip() 
        if example!="...":
            html += "<tr>" 
            for i in range(len(example.split("|"))):
                if example.split("|")[i].strip()!="...":
                    html += "<td>" + example.split("|")[i].strip() + "</td>" 
            
            html += "</tr>\n" 
    html += "</table>\n" # 作者和时间部分 
    html += "<p><b>作者：</b>" + text[text.index("*作者：*") + 5:text.index("*创建时间：*")].strip().split('\n')[0] + "</p>\n"
    html += "<p><b>创建时间：</b>" + text[text.index("*创建时间：*") + 7:text.index("*最后修改时间：*")].strip().split('\n')[0] + "</p>\n"
    html += "<p><b>最后修改时间：</b>"  + text[text.index("*最后修改时间：*") + 9:].strip().split('\n')[0] + "</p>\n"
    
    return html

#@cache_page(60 * 15)
def keyword_search(request): 
    start_time = time.time()
    model_status_list=[]
    with open('/var/www/LRM/model_status.json', 'r') as json_file:
        model_status_dict = json.load(json_file)
    for k,v in model_status_dict.items():
        model_status_list.append(v)
    if request.method == 'POST': 
        keyword = request.POST.get('keyword')
        keyword = keyword.strip()
        db_name='中台关键字库'
        #db_name =  request.POST.getlist('db_name')
        #choice = request.POST.get('choice')
        choice = 'no'
        output_number = request.POST.get('output_number')
        #keyword_type = request.POST.get('keyword_type')
        keyword_type = 'all'
        #keyword_search_range = request.POST.get('keyword_search_range')
        keyword_search_range = 'yes'
        ai_enable = 'ai-enable'
        #ai_enable = request.POST.get('ai_enable')
        model = request.POST.get('model')
        temperature = request.POST.get('temperature')
        user_ip = request.META.get('REMOTE_ADDR')
        db_list= '、'.join(db_name) 
        db_name_list=[]
        #cache_key = f'keyword_search_{keyword}_{db_name}_{choice}_{keyword_type}_{keyword_search_range}_{ai_enable}'
        #cached_data = cache.get(cache_key)
        if ai_enable=='ai-enable':
            if keyword.startswith('!<') or keyword.startswith('show '):
                config_search_obj = ConfigSearch(config_search_item=keyword, user_ip=user_ip)
                config_search_obj.save()
        else:
            if "!" not in keyword:
                keyword_search_obj = KeywordSearch(keyword_search_item=keyword, user_ip=user_ip)
                keyword_search_obj.save()
        # if cached_data:
        #     if ai_enable=='ai-enable':
        #         return render(request, 'search_results_with_config.html', cached_data)
        #     else:
        #         if choice=='yes':
        #             return render(request, 'search_results_without_example.html', cached_data)
        #         elif choice=='no':
        #             return render(request, 'search_results_without_example_all.html', cached_data)

        # else:
        for db_name_item in db_name:
            if db_name_item.split('关键字库')[0]=='中台':
                db_name_list.append('keywords')
                db_name_list.append('lib')
            elif db_name_item.split('关键字库')[0]=='15K':
                #db_name_list.append('keywords_9k')
                db_name_list.append('zxr15k')
            elif db_name_item.split('关键字库')[0]=='C89E':
                db_name_list.append('keywords_c89e')
            elif db_name_item.split('关键字库')[0]=='UFP':
                db_name_list.append('keywords_ufp')
                db_name_list.append('zxrUFP')
            elif db_name_item.split('关键字库')[0]=='DC':
                db_name_list.append('keywords_dc')
                db_name_list.append('zxrdc')
                db_name_list.append('keywords_9k')
                
            else:
                db_name_list.append('keywords_%s' % db_name_item.split('关键字库')[0].lower())
        matches=[]
        query_list = []
        if '|' in keyword: 
            query_list = [keyword.strip() for keyword in keyword.split('|')]
        elif '&' in keyword:
            query_list = [keyword.strip() for keyword in keyword.split('&')]
        else:
            query_list = [keyword.strip()]
        #lines_html = """<p><b>参数介绍：</b></p>"""
        #lines_html = convert_to_html(line_text)
        keyword_list=[]
        keyword_lower=keyword.lower()
        results=[]
        script_results=[]
        script_results_all=[]
        sample_number = 0
        # for keyword_item in KeywordList.objects.all():
        #     keyword_item_lower=keyword_item.keyword_name.lower()
        #     if keyword_type == 'all':
        #         if keyword_lower in keyword_item_lower:
        #             if keyword_item.keyword_name not in keyword_list:
        #                 keyword_list.append(keyword_item.keyword_name)
        #     elif keyword_type == '配置':
        #         if (keyword_lower in keyword_item_lower) and ((re.match('^加%s' % keyword_type , keyword_item_lower)) or (re.match('^删%s' % keyword_type , keyword_item_lower)) or (re.match('^%s' % keyword_type , keyword_item_lower))):
        #             if keyword_item.keyword_name not in keyword_list:
        #                 keyword_list.append(keyword_item.keyword_name)
        #     else:
        #         if (keyword_lower in keyword_item_lower) and (re.match('^%s' % keyword_type , keyword_item_lower)):
        #             if keyword_item.keyword_name not in keyword_list:
        #                 keyword_list.append(keyword_item.keyword_name)
        #keyword_list_query, config_dict_query = query_data(keyword_lower,keyword_search_range,ai_enable)

        # keyword_list_query, config_dict_query = query_data.delay(keyword_lower,keyword_search_range,ai_enable)
        ip = get_ip_address()
        if ai_enable=='ai-enable':
            if keyword_lower.startswith('show '):
                show_keyword_list, show_command_list, execution_time = chat_knowledge_for_show_command(keyword_lower,'h11-23-show', top_k=output_number)
                show_keyword_list_all = []
                show_keyword_option_list_all = []
                show_keyword_get_info_option_list_all = []
                show_command_list_all = [element for element in show_command_list for _ in range(3)]
                for item in show_keyword_list:
                    obj=KeywordDoc.objects.filter(keyword_name=item)
                    get_info_option_list=[]
                    if len(obj)==1:
                        soup = BeautifulSoup(obj[0].keyword_doc_html, 'html.parser')
                        tables = soup.find_all('table')
                        english_only_regex = re.compile(r'^[a-zA-Z_]+$')
                        if len(tables)>=1:
                            rows = tables[0].find_all('tr')
                            for row in rows:
                                first_column = row.find('td')
                                if first_column:
                                    if english_only_regex.match(first_column.text):
                                        get_info_option_list.append(first_column.text)
                        option_list = re.findall(r'<p><code>(.*?)</code>', obj[0].keyword_doc_html)
                    else:
                        option_list = []
                    show_keyword_list_all.append(item)
                    show_keyword_option_list_all.append(option_list)
                    show_keyword_get_info_option_list_all.append(get_info_option_list)
                    item=item.replace('获取','检查')
                    show_keyword_list_all.append(item)
                    show_keyword_option_list_all.append(option_list)
                    show_keyword_get_info_option_list_all.append(get_info_option_list)
                    item=item.replace('检查','重复检查')
                    show_keyword_list_all.append(item)
                    show_keyword_option_list_all.append(option_list)
                    show_keyword_get_info_option_list_all.append(get_info_option_list)
                results=[]
                for i in range(len(show_keyword_list_all)):
                #for keyword_name in show_keyword_list_all:
                    rows=[]
                    #queryset = KeywordDoc.objects.filter(keyword_name='%s' % keyword_name)
                    queryset = KeywordDoc.objects.filter(keyword_name='%s' % show_keyword_list_all[i])
                    if len(queryset)>=1:
                        if queryset[0].keyword_doc_html:
                            lines = queryset[0].keyword_doc_html.split('\n')
                            #rows.append(keyword_name)
                            rows.append(show_keyword_list_all[i])
                            rows.append(lines)
                            rows.append(show_command_list_all[i])
                            rows.append(show_keyword_option_list_all[i])
                            rows.append(show_keyword_get_info_option_list_all[i])
                            results.append(rows)
                    else:
                        lines = ''
                        rows.append('')
                        rows.append(lines)
                        rows.append(show_command_list_all[i])
                        rows.append('')
                        rows.append('')
                        results.append(rows)
                total_number=len(results)
                return render(request, 'search_results_for_show_command.html', {'results': results, 'total_number': total_number, 'ip': ip, 'run_time': execution_time, 'show_command': keyword_lower,'show_command_list': show_command_list})
                #return HttpResponse("%s %s %s" % (keyword_lower,show_keyword_list,execution_time))
            elif keyword_lower.startswith('!<'):
                query_data_task = query_data.delay(keyword_lower,keyword_search_range,ai_enable,model,temperature)
                task_id = query_data_task.task_id
                #print(f'Celery Task ID: {task_id}')
                result = AsyncResult(task_id)
                if result:
                    return render(request, 'search_results_link_with_task_id.html', {'ip': ip, 'task_id': task_id,'result': result})
                else:
                    return render(request, 'search_no_results_link_with_task_id.html', {'ip': ip, 'task_id': task_id})
            elif '测试仪' in keyword_lower:
                ceshiyi_keyword_testcase_list, execution_time = chat_knowledge_for_ceshiyi_keyword_testcase(keyword,'ceshiyi_keyword_capi', top_k=output_number)
                results=[]
                for i in range(len(ceshiyi_keyword_testcase_list)):
                    rows=[]
                    lines=ceshiyi_keyword_testcase_list[i].split('\n')
                    ceshiyi_keyword_list=[]
                    for item in lines:
                        item=item.strip()
                        ceshiyi_keyword_list.append(item.split('  ')[0])
                    lines_for_html=[]
                    for item in lines:
                        item=item.replace('    ', '&nbsp;&nbsp;&nbsp;&nbsp;')
                        lines_for_html.append(item)
                    rows.append(lines_for_html)
                    rows.append(ceshiyi_keyword_list)
                    ceshiyi_keyword_doc_list=[]
                    for keyword_name in ceshiyi_keyword_list:
                        queryset = KeywordDoc.objects.filter(keyword_name='%s' % keyword_name)
                        if len(queryset)>=1:
                            if queryset[0].keyword_doc_html:
                                doc_lines = queryset[0].keyword_doc_html.split('\n')
                                ceshiyi_keyword_doc_list.append([keyword_name,doc_lines])
                        else:
                            doc_lines=''
                            ceshiyi_keyword_doc_list.append([keyword_name,doc_lines])
                    rows.append(ceshiyi_keyword_doc_list)
                    results.append(rows)
                total_number=len(results)
                return render(request, 'search_results_for_ceshiyi_keyword_testcase.html', {'results': results, 'total_number': total_number, 'ip': ip, 'run_time': execution_time, 'keyword': keyword})
                #return HttpResponse("%s %s" % (ceshiyi_keyword_testcase_list,execution_time))
            else:
                keyword_name_list, execution_time = chat_knowledge_for_keyword_doc(keyword_lower,'keyword_doc_20231218', top_k=output_number)
                results=[]
                for i in range(len(keyword_name_list)):
                    rows=[]
                    queryset = KeywordDoc.objects.filter(keyword_name='%s' % keyword_name_list[i])
                    if len(queryset)>=1:
                        if queryset[0].keyword_doc_html:
                            lines = queryset[0].keyword_doc_html.split('\n')
                            rows.append(keyword_name_list[i])
                            rows.append(lines)
                            results.append(rows)
                    else:
                        lines = ''
                        rows.append('')
                        rows.append(lines)
                        results.append(rows)
                total_number=len(results)
                return render(request, 'search_results_for_keyword_doc.html', {'results': results, 'total_number': total_number, 'ip': ip, 'run_time': execution_time, 'keyword_name': keyword_lower,})
                #return HttpResponse("模糊推荐关键字%s" % keyword_name_list)
                #return HttpResponse("请输入正确格式的配置命令或show命令行")
        else:
            keyword_list_query, config_dict_query, run_time = query_data(keyword_lower,keyword_search_range,ai_enable,model,temperature)
        #time.sleep(90)
        # if result.ready():
        #     print(result.get())
        # else:
        #     print("任务尚未完成")
        # while result.status!='SUCCESS':
        #     print('status=%s' % result.status)
        # print(result.get())
        # return render(request, 'celery_bar.html', {'task_id': task_id})
        #return HttpResponse("Celery works: %s" % result.result)
        #return HttpResponse("Celery works: %s %s" % (result.result[0],result.result[1]))
        # keyword_list_query=result.result[0]
        # config_dict_query=result.result[1]
        for keyword_item in keyword_list_query:
            keyword_item_lower=keyword_item.lower()
            if keyword_type == 'all':
                keyword_list.append(keyword_item)
            elif keyword_type == '配置':
                if (re.match('^加%s' % keyword_type , keyword_item_lower)) or (re.match('^删%s' % keyword_type , keyword_item_lower)) or (re.match('^%s' % keyword_type , keyword_item_lower)):
                    keyword_list.append(keyword_item)
            elif keyword_type == '检查':
                if (re.match('^%s' % keyword_type , keyword_item_lower)) or (re.match('^获取' , keyword_item_lower)) or (re.match('^重复%s' % keyword_type, keyword_item_lower)):
                    keyword_list.append(keyword_item)
            else:
                if (re.match('^%s' % keyword_type , keyword_item_lower)):
                    keyword_list.append(keyword_item)
        keyword_list_sorted = sorted(keyword_list, key=len, reverse=True)
        keyword_doc_dict = {}
        for keyword_name in keyword_list_sorted:
            #pattern = r'%s\n(.*?)\[Arguments\]' % keyword_name
            #for item in KeywordPost.objects.all():
            lines=''
            rows=[]
            #line_text = item.keyword
            #result = re.findall(pattern, item.keyword, re.DOTALL)
                
                #if result:
            queryset = KeywordDoc.objects.filter(keyword_name='%s' % keyword_name)
            if queryset[0].keyword_loc:
                if queryset[0].keyword_loc.split('/')[3].lower() in db_name_list:
                    #line_text = result[0]
                    #lines = result[0].split('...')
                    queryset = KeywordDoc.objects.filter(keyword_name='%s' % keyword_name)
                    #if queryset[0].keyword_doc:
                    if queryset[0].keyword_doc_html:
                        #replace_str = '&lt;mark&gt;%s&lt;/mark&gt' % keyword
                        #highlighted_keyword_doc = queryset[0].keyword_doc.replace(keyword, replace_str)
                        #lines = highlighted_keyword_doc.split('...')
                        #lines = queryset[0].keyword_doc.split('...')
                        highlighted_lines=queryset[0].keyword_doc_html
                        for query_item in query_list: 
                            pattern = query_item
                            match = re.search(pattern, highlighted_lines, flags=re.IGNORECASE) 
                            if match: 
                                matched_string = match.group()
                                highlighted_lines = highlighted_lines.replace(matched_string, f"<span class='highlight'>{matched_string}</span>") 
                            else:
                                highlighted_lines = highlighted_lines.replace(query_item, f"<span class='highlight'>{query_item}</span>")
                        
                        # pattern = keyword.strip()
                        # match = re.search(pattern, queryset[0].keyword_doc_html, flags=re.IGNORECASE) 
                        # if match: 
                        #     matched_string = match.group()
                        #     highlighted_lines = queryset[0].keyword_doc_html.replace(matched_string, f"<span class='highlight'>{matched_string}</span>") 
                        # else:
                        #     highlighted_lines = queryset[0].keyword_doc_html.replace(keyword, f"<span class='highlight'>{keyword}</span>")

                        #highlighted_lines = queryset[0].keyword_doc_html.replace(keyword, f"<span class='highlight'>{keyword}</span>")
                        lines = highlighted_lines.split('\n')
                        #lines = queryset[0].keyword_doc_html.split('\n')
                    #line_text = '###\n'.join(lines)
                    #lines_html = convert_to_html(lines)
                    
                    #script_name_list=[]
                    for query_item in query_list:
                        
                        pattern = query_item
                        match = re.search(pattern, keyword_name, flags=re.IGNORECASE) 
                        if match: 
                            matched_string = match.group()
                            keyword_name_highlight = keyword_name.replace(matched_string, f"<span class='highlight'>{matched_string}</span>") 
                            
                        else:
                            keyword_name_highlight = keyword_name.replace(query_item, f"<span class='highlight'>{query_item}</span>")
                    script_results_all=[]
                    rows.append(keyword_name_highlight)
                    rows.append(lines)
                    rows.append(keyword_name)
                    if ai_enable=='ai-enable':
                        #rows.append(keyword_dict_query['%s' % keyword_name].replace('\n', '<br>'))
                        keyword_doc_dict['%s' % keyword_name]=[keyword_name_highlight, lines]
                    results.append(rows)
                    # if choice=='yes':
                    #     for script_item in ScriptPost.objects.all():
                    #         script_item_list = script_item.script.split('\n')
                    #         #for line in script_item_list:
                            
                    #         if script_item.script_location+script_item.script_name not in script_name_list:
                    #             script_name_list.append(script_item.script_location+script_item.script_name)
                    #             for i in range(len(script_item_list)):   
                    #                 #if  keyword_name.lower() in line.lower():
                    #                 if ' %s ' % keyword_name.lower() in script_item_list[i].lower():
                                        
                    #                         if script_item_list[i] not in script_results:
                    #                             #script_results.append('%s [示例索引：%s/%s.robot]' % (line,script_item.script_location,script_item.script_name))
                    #                             script_with_title=[]
                    #                             script_results.append(script_item_list[i])
                                                
                    #                             script_with_title.append('[示例索引号 %s ：%s/%s.robot]' % (script_item.id,script_item.script_location,script_item.script_name))
                    #                             script_with_title.append(script_item.id)
                    #                             script_with_title.append(script_item_list[i-5:i+6])
                    #                             script_results_all.append(script_with_title)
                    # if choice=='yes':
                    #     if queryset[0].keyword_example:
                    #         # example_content = queryset[0].keyword_example.replace('\t', '    ')
                    #         # example_dic = eval(example_content)
                    #         example_dic = eval(queryset[0].keyword_example)
                    
                    #     #id_list=[]
                    #         for key,value in example_dic.items():
                    #             script_with_title=[]
                    #             new_string_list=[]
                    #             #if value[1] not in id_list:
                    #             html_content = highlight('\n'.join(value[2]), RobotFrameworkLexer(), HtmlFormatter(full=True))
                    #             html_content_list = html_content.split('\n')[98:109]
                    #             for item in html_content_list:
                    #             #for item in value[2]:
                    #                 item = re.sub(r'^\t', '    ', item)
                    #                 for query_item in query_list: 
                    #                     pattern = query_item
                    #                     match = re.search(pattern, item, flags=re.IGNORECASE) 
                    #                     if match: 
                    #                         matched_string = match.group()
                    #                         item = item.replace(matched_string, f"<span class='highlight'>{matched_string}</span>")
                    #                         #item = item.replace(item, f"<span class='highlight'>{item}</span>")
                    #                     else:
                    #                         item = item.replace(query_item, f"<span class='highlight'>{query_item}</span>")
                                            
                    #                 new_string_list.append(item)
                                
                    #             script_with_title.append(value[0])
                    #             script_with_title.append(value[1])
                    #             script_with_title.append(new_string_list)
                    #             #script_with_title.append(value[2])
                    #             script_results_all.append(script_with_title)
                    #             #id_list.append(value[1])
                    # elif choice=='no':
                    #     if queryset[0].testcase_example:
                    #         example_dic = eval(queryset[0].testcase_example)
                    #     #else:
                    #     #    example_dic = {}
                    #         id_list=[]
                    #         for key,value in example_dic.items():
                    #             script_with_title=[]
                    #             new_string_list=[]
                    #             if value[1] not in id_list:
                    #                 for item in value[2]:
                    #                     item = re.sub(r'^\t', '    ', item)
                    #                     for query_item in query_list: 
                    #                         pattern = query_item
                    #                         match = re.search(pattern, item, flags=re.IGNORECASE) 
                    #                         if match: 
                    #                             matched_string = match.group()
                    #                             item = item.replace(matched_string, f"<span class='highlight'>{matched_string}</span>")
                    #                             #item = item.replace(item, f"<span class='highlight'>{item}</span>")
                    #                         else:
                    #                             item = item.replace(query_item, f"<span class='highlight'>{query_item}</span>")
                                                
                    #                     new_string_list.append(item)
                                    
                    #                 script_with_title.append(value[0])
                    #                 script_with_title.append(value[1])
                    #                 script_with_title.append(new_string_list)
                    #                 #script_with_title.append(value[2])
                    #                 script_results_all.append(script_with_title)
                    #                 id_list.append(value[1])
                                
                    # # for query_item in query_list:  
                    # #     if query_item in keyword_name:
                    # #         highlighted_keyword_name = keyword_name.replace(query_item, f"<span class='highlight'>{query_item}</span>")
                    # #     else:
                    # #         highlighted_keyword_name = keyword_name
                    # #     highlighted_keyword_name = re.sub(re.escape(query_item), f"<span class='highlight'>{query_item}</span>", keyword_name, flags=re.IGNORECASE)
                    # for query_item in query_list:
                        
                    #     pattern = query_item
                    #     match = re.search(pattern, keyword_name, flags=re.IGNORECASE) 
                    #     if match: 
                    #         matched_string = match.group()
                    #         keyword_name_highlight = keyword_name.replace(matched_string, f"<span class='highlight'>{matched_string}</span>") 
                            
                    #     else:
                    #         keyword_name_highlight = keyword_name.replace(query_item, f"<span class='highlight'>{query_item}</span>")
                            
                    
                    # # pattern = keyword.strip()
                    # # match = re.search(pattern, keyword_name, flags=re.IGNORECASE) 
                    # # if match: 
                    # #     matched_string = match.group()
                    # #     highlighted_keyword_name = keyword_name.replace(matched_string, f"<span class='highlight'>{matched_string}</span>") 
                    # # else:
                    # #     highlighted_keyword_name = keyword_name.replace(keyword, f"<span class='highlight'>{keyword}</span>")
                    # #rows.append(highlighted_keyword_name)
                    # rows.append(keyword_name_highlight)
                    # rows.append(lines)
                    # script_results_all_dict = {index+1: sample for index, sample in enumerate(script_results_all)} 
                    # rows.append(script_results_all_dict)
                    # sample_number = sample_number + len(script_results_all)
                    # rows.append(keyword_name)
                    # results.append(rows)
                    # #         rows.append(script_results_all_dict)
                    # if (choice=='yes') or (choice=='all'):
                    #     #if script_results_all:
                    #     if output_number=='3':
                    #         if len(script_results_all)>=3:
                    #             script_sample=[script_results_all[0],script_results_all[len(script_results_all)//2],script_results_all[-1]]
                    #             script_sample_dict = {index+1: sample for index, sample in enumerate(script_sample)} 
                    #             rows.append(script_sample_dict)
                                
                    #         else:
                    #             script_results_all_dict = {index+1: sample for index, sample in enumerate(script_results_all)} 
                    #             rows.append(script_results_all_dict)
                            
                    #     elif output_number=='5':
                    #         if len(script_results_all)>=5:
                    #             script_sample=[script_results_all[0],script_results_all[len(script_results_all)//2//2],script_results_all[len(script_results_all)//2],script_results_all[-len(script_results_all)//2//2],script_results_all[-1]]
                    #             script_sample_dict = {index+1: sample for index, sample in enumerate(script_sample)} 
                    #             rows.append(script_sample_dict)
                    #         else:
                    #             script_results_all_dict = {index+1: sample for index, sample in enumerate(script_results_all)} 
                    #             rows.append(script_results_all_dict)
                    #     elif output_number=='7':
                    #         if len(script_results_all)>=9:
                    #             script_sample=[script_results_all[0],script_results_all[len(script_results_all)//2//2//2],script_results_all[len(script_results_all)//2//2],script_results_all[len(script_results_all)//2],script_results_all[-len(script_results_all)//2//2],script_results_all[-len(script_results_all)//2//2//2],script_results_all[-1]]
                    #             script_sample_dict = {index+1: sample for index, sample in enumerate(script_sample)} 
                    #             rows.append(script_sample_dict)
                    #         else:
                    #             script_results_all_dict = {index+1: sample for index, sample in enumerate(script_results_all[0:7])}
                    #             rows.append(script_results_all_dict)
                    #     elif output_number=='17':
                    #         if len(script_results_all)>=17:
                    #             script_sample=[script_results_all[0],script_results_all[len(script_results_all)//2//2//2//2],script_results_all[len(script_results_all)//2//2//2],script_results_all[len(script_results_all)//2//2],script_results_all[len(script_results_all)//2],script_results_all[-len(script_results_all)//2//2],script_results_all[-len(script_results_all)//2//2//2],script_results_all[-len(script_results_all)//2//2//2//2],script_results_all[-1]]
                    #             script_sample_dict = {index+1: sample for index, sample in enumerate(script_sample)} 
                    #             rows.append(script_sample_dict)
                    #         else:
                    #             script_results_all_dict = {index+1: sample for index, sample in enumerate(script_results_all[0:9])}
                    #             rows.append(script_results_all_dict)
                    #     elif output_number=='all':
                    #         script_results_all_dict = {index+1: sample for index, sample in enumerate(script_results_all)} 
                    #         rows.append(script_results_all_dict)
                                
                    #         #results.append(rows)
                    #     #if script_results_all:
                    #     sample_number = sample_number + len(script_results_all)
                    #     rows.append(keyword_name)
                    #     results.append(rows)
                    # else:
                    #     rows.append(keyword_name)
                    #     results.append(rows)   

            total_number = len(results) 
            end_time = time.time()
            run_time = round(end_time - start_time, 1)
            results_invert = results[::-1]
            
            #ip = '10.57.170.40'
            #cache.set(cache_key, {'results': results_invert, 'total_number': total_number, 'db_list': db_list, 'sample_number': sample_number, 'run_time': run_time, 'ip': ip})

            # if choice=='yes':
                 
            #     return render(request, 'search_results.html', {'results': results, 'total_number': total_number, 'db_list': db_list, 'sample_number': sample_number, 'run_time': run_time})

            # elif choice=='all':
                
            #     return render(request, 'search_results_all.html', {'results': results, 'total_number': total_number, 'db_list': db_list, 'sample_number': sample_number, 'run_time': run_time})
            # else:
                
            #     return render(request, 'search_results_without_example.html', {'results': results, 'total_number': total_number, 'db_list': db_list,'sample_number': sample_number, 'run_time': run_time})

            # if ai_enable=='ai-enable':
            #     results_invert = []
            #     for k,v in config_dict_query.items():
            #         config_rows = []
            #         # config_rows.append(keyword_doc_dict[v][0])
            #         # config_rows.append(keyword_doc_dict[v][1])
            #         # config_rows.append(v)
            #         config_rows.append(keyword_doc_dict[v[0]][0])
            #         config_rows.append(keyword_doc_dict[v[0]][1])
            #         config_rows.append(v[0])
            #         k=k.replace('<', '&lt;')
            #         k=k.replace('>', '&gt;')
            #         k=k.replace(' ', '&nbsp;')
            #         k=k.replace('\n', '<br>')
            #         config_rows.append(k)
            #         v[1]=v[1].replace(' ', '&nbsp;')
            #         config_rows.append(v[1])
            #         results_invert.append(config_rows)

            #     cache.set(cache_key, {'results': results_invert, 'total_number': total_number, 'db_list': db_list, 'sample_number': sample_number, 'run_time': run_time, 'ip': ip})

            #     #task = query_data.delay(keyword,keyword_search_range,ai_enable)
            #     #return render(request, 'progress.html', {'task_id': task.id})
            #     return render(request, 'search_results_with_config.html', {'results': results_invert, 'total_number': total_number, 'db_list': db_list,'sample_number': sample_number, 'run_time': run_time, 'ip': ip})

            # else:
            #cache.set(cache_key, {'results': results_invert, 'total_number': total_number, 'db_list': db_list, 'sample_number': sample_number, 'run_time': run_time, 'ip': ip})

            if choice=='yes':
                
                return render(request, 'search_results_without_example.html', {'results': results_invert, 'total_number': total_number, 'db_list': db_list,'sample_number': sample_number, 'run_time': run_time, 'ip': ip})

            elif choice=='no':
                
                return render(request, 'search_results_without_example_all.html', {'results': results_invert, 'total_number': total_number, 'db_list': db_list,'sample_number': sample_number, 'run_time': run_time, 'ip': ip})

        
        
        
    else: 
        total_keyword_number = len(KeywordDoc.objects.all())
        # total_keyword_number = len(KeywordList.objects.all())
        total_file_number = len(KeywordPost.objects.all())
        # total_script_number = len(ScriptPost.objects.all())
        #return render(request, 'search_form.html', {'total_keyword_number': total_keyword_number, 'total_file_number': total_file_number, 'total_script_number': total_script_number}) 
        #sorted_search_keys = get_sorted_search_keys()
        ip = get_ip_address()
        #ip = '10.57.170.40'
        user_ip = request.META.get('REMOTE_ADDR')
        #sorted_search_keys, search_counter_today, search_counter_all, user_ip_counter = get_sorted_search_keys()
        sorted_search_keys, search_counter_today, search_counter_all, user_ip_counter, sorted_search_config = get_sorted_search_keys()
        #return render(request, 'search_form.html', {'sorted_search_keys': sorted_search_keys[:50]})
        #return render(request, 'search_form.html', {'sorted_search_keys': sorted_search_keys[:50], 'search_counter_today': search_counter_today, 'search_counter_all': search_counter_all, 'user_ip': user_ip, 'ip': ip, 'user_ip_counter': user_ip_counter })
        return render(request, 'search_form.html', {'sorted_search_keys': sorted_search_keys[:50], 'search_counter_today': search_counter_today, 'search_counter_all': search_counter_all, 'user_ip': user_ip, 'ip': ip, 'user_ip_counter': user_ip_counter, 'sorted_search_config': sorted_search_config[:50], 'model_status_list': model_status_list})
        #return render(request, 'search_form.html') 
        # keyword_number_list = [0,0,0,0,0]
        # for item in KeywordList.objects.all():
        #     print(item.keyword_loc)
        #     if item.keyword_loc:
        #         if '/root/rfhub/keywords/' in item.keyword_loc:
        #             keyword_number_list[0]=keyword_number_list[0]+1
        #         elif '/root/rfhub/keywords_9k/' in item.keyword_loc:
        #             keyword_number_list[1]=keyword_number_list[1]+1
        #         elif '/root/rfhub/keywords_c89e/' in item.keyword_loc:
        #             keyword_number_list[2]=keyword_number_list[2]+1
        #         elif '/root/rfhub/keywords_UFP/' in item.keyword_loc:
        #             keyword_number_list[3]=keyword_number_list[3]+1
        #     else:
        #         keyword_number_list[4]=keyword_number_list[4]+1
                
        # return render(request, 'search_form.html', {'keyword_number_list': keyword_number_list}) 
        #return render(request, 'search_form.html')

@csrf_exempt  # 如果您在开发过程中遇到CSRF验证问题，可以暂时使用此装饰器
def execute_code(request):
    if request.method == 'POST':
        print('返回结果：%s' % request.body)  # 打印请求体内容，看看实际接收到的数据
        try:
            # 从request.body中读取原始JSON数据
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            # 如果JSON解析出错，返回错误信息
            return JsonResponse({'error': str(e)}, status=400)

        # 从解析后的数据中获取'code'字段的值
        code = data.get('code', '')

        # 在这里可以添加代码执行逻辑
        # ...

        # 返回执行结果，这里暂时只是将接收到的代码返回
        _autolearn = AutoLearn()
        # configtext = _autolearn.dryrun_keyword(keywordtext=code)
        
        # code_list=code.split('\n')
        # text_list=[]
        # for item in code_list:
        #     if item:
        #         item=item.replace('&nbsp;',' ')
        #         test_list.append(item.strip())
        # text_content='\n'.join(text_list)
        code=code.replace('\xa0',' ')
        configtext = _autolearn.dryrun_keyword_text(text=code)
        configtext = json.loads(configtext)
        results_list=[]
        for item in configtext:
            if item['status']==1:
                #results_list.append('关键字代码：\n')
                results_list.append('关键字代码：%s\n' % item['keywordtext'])
                results_list.append('执行结果：\n')
                results_list.append('%s\n' % item['result'])
            else:
                #results_list.append('关键字代码：\n')
                results_list.append('关键字代码：%s\n' % item['keywordtext'])
                results_list.append('报错信息：\n')
                results_list.append('%s\n' % item['result'])
                #results_list.append('关键字无法执行，请检查关键字名称或参数是否正确\n')
        results='\n'.join(results_list)
        return HttpResponse(results)
    else:
        # 如果不是POST请求，渲染并返回HTML页面
        return render(request, 'execute_code.html')

@csrf_exempt  # 如果您在开发过程中遇到CSRF验证问题，可以暂时使用此装饰器
def chat_api(request):
    if request.method == 'POST':
        model = 'Qwen-72B-Chat'
        temperature = 0.5

        # prefix = '''
        # # 角色：资深客服代表
        # '''
        prefix = '''
        你的角色是资深的网络流量开发工程师，辅助用户通过用Robot Framework开发的测试仪关键字设计网络流量自动收集参数信息。
        请根据下面的步骤一步一步根据指定文档的信息解决用户的问题：
        第一步：你要首先需要仔细阅读测试仪关键字分类列表中的关键字名称和关键字用途，确保了解了每个关键字的用途。
        第二步：根据用户输入的内容一步一步确认用户的需求，然后根据用户的需求推荐关键字名称。
        第三步：等待用户回复收集网络流量的相关信息。收集完信息需确认用户是否还需要添加其他内容。
        最后需要询问是否需要结束本轮对话，告诉用户通过多轮对话收集到的信息的汇总结果。
        
        注意事项：
        请确保明确用户的真实需求、网络流量所需要的信息和附加要求，以便从说明文档中识别出该项唯一的内容。
        你的回应应该以简短、言简意赅和友好的风格呈现。

        测试仪关键字分类列表：

        配置类关键字：
        关键字名称                          关键字用途
        测试仪配置流量                       1. 在测试仪物理端口上配置一条流量
        测试仪配置Traffic Profile           1. 在测试仪物理端口上配置一个Traffic Profile，端口上的多条流可以平均分配这个Traffic Profile的Load 注意一个端口只配置一个Traffic Profile，配置多个时Load会有问题 2. 如果没有创建Profile，缺省有一个Profile1,名字也为portName_Profile1
        测试仪配置Tag IPv4 Host              1. 在测试仪物理端口上配置一条Tag的IPv4 Host
        测试仪配置Tag IPv6 Host              1. 在测试仪物理端口上配置一条Tag的IPv6 Host
        测试仪配置UnTag IPv6 Host            1. 在测试仪物理端口上配置一条UnTag的IPv6 Host 需求来源(链接)： 任务名称： RF关键字实现使用testcenter测试仪模拟host，使能mld路由器，支持mld加入、离开

        检查类关键字：
        关键字名称                          关键字用途
        测试仪检查发流包数
        测试仪获取端口统计
        测试仪获取流量统计的发送total_pkts 
        测试仪获取流量统计的接收Frames
        测试仪获取端口统计的接收L1BitRate
        测试仪获取流量统计的发送Bytes
        '''
        PREFIX=prefix
        question='配置网络流量'
        answer='''
        {
        "配置类关键字": "测试仪配置流量",
        }
        '''
        
        examples=[]
        examples.append({"question": question,"answer": answer,})

        example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        OpenAIModel(model_name='m3e-base'),
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        FAISS,
        # This is the number of examples to produce.
        k=5
        )

        example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
            ("ai", "{answer}"),
        ]
        )
        chat_prompt_template_prompt = FewShotChatMessagePromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        input_variables=["input"],
        )

        final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PREFIX),
            chat_prompt_template_prompt,
            ("human", "{input}"),
        ]
        )

        chat = JuLLMChatChat(temperature=temperature, model=model)
        chain = LLMChain(llm=chat, prompt=final_prompt, verbose=False)
        try:
            message = request.body.decode('utf-8')  # 确保正确解码请求体
            question=json.loads(request.body)  # 尝试解析JSON数据
            # ... 处理msg_new中的数据 ...
            #return JsonResponse({'status': 'success', 'data': message})
        except json.JSONDecodeError as e:
        # 如果JSON解析失败，返回错误信息
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON request.body: %s' % request.body}, status=400)
        #question=json.loads(request.body)
        # history_message=question['chatLog'].replace('</p><p>','\n')
        # history_message=history_message.replace('<p>','')
        # history_message=history_message.replace('</p>','')
        delimiter = "####"
        system_message = """
        """
        # system_message ="""
        # 你将提供服务查询。
        # 服务查询将使用"####"字符分隔。

        # 仅输出一个 Python 对象列表，其中每个对象具有以下格式：
        #     'get_info': <必须在下面的说明文档中找到get_info参数列表里面的参数值>

        # 请仔细阅读说明文档中get_info参数列表第一列的参数名称和第二列的参数说明
        # 如果用户提及了参数说明中的内容，则必须将其与参数列表中的正确参数值相关联。
        # 如果根据用户输入的信息未找到get_info列表里对应的参数值，则输出空列表。
        
        # HTML格式的说明文档：
        # '''
        # <p>获取IP PIM Interface Brief信息</p>
        # <p>方式一：指定 <code>output</code> ，将从 <code>output</code> 内容中提取数据</p>
        # <p>方式二：指定 <code>device</code> ，将在设备上执行 show ip pim interface brief后在回显中获取 ip pim 信息</p>
        # <p><b>参数介绍：</b></p>
        # <p><code>output</code>    指定回显内容，若不指定output，必须指定 <code>device</code></p>
        # <p><code>device</code>    指定设备</p>
        # <p><code>get_info</code>    指定要获取的信息(必填)， <code>get_info</code> 支持的参数如下:</p>
        # <table border="1">
        # <tr>
        # <td>total</td>
        # <td>获取接口总数</td>
        # </tr>
        # <tr>
        # <td>interface</td>
        # <td>获取接口名称</td>
        # </tr>
        # <tr>
        # <td>state</td>
        # <td>获取接口状态up/down</td>
        # </tr>
        # <tr>
        # <td>nbr_count</td>
        # <td>获取邻居个数</td>
        # </tr>
        # <tr>
        # <td>hello_period</td>
        # <td>获取Hello报文的发送时间间隔</td>
        # </tr>
        # <tr>
        # <td>dr_priority</td>
        # <td>获取该接口的DR优先级</td>
        # </tr>
        # <tr>
        # <td>dr_address</td>
        # <td>获取该接口的DR地址</td>
        # </tr>
        # </table>
        # <p><b>示例：</b></p>
        # <table border="1">
        # <tr>
        # <td><a href="#%E8%8E%B7%E5%8F%96IP%20PIM%20Interface%20Brief%E4%BF%A1%E6%81%AF" class="name">获取IP PIM Interface Brief信息</a></td>
        # <td>device=${zxr0}</td>
        # <td>interface=vlan1</td>
        # <td>get_info=dr_priority</td>
        # </tr>
        # </table>
        # <p><b>作者：</b> 时瑞研</p>
        # <p><b>创建时间：</b> 2022-03-15</p>
        # <p><b>最后修改时间：</b> 2022-03-15</p>
        # '''

        # 仅输出 Python 对象列表，不包含其他字符信息。
        # """
        
        user_message = f"""\ 
        {question['message']}"""
        messages =  [  
        {'role':'system', 
        'content': system_message},    
        {'role':'user', 
        'content': f"{delimiter}{user_message}{delimiter}"},  
        ]

        cmd='%s\n%s' % (question['chatLog'],messages)
        #cmd='%s' % (messages)
        
        #cmd='%s\n%s' % (question['chatLog'],question['message'])
        print('输入内容：%s' % cmd)
        msg=chain.run(cmd)
        print('返回结果：%s' % msg)  # 打印请求体内容，看看实际接收到的数据
        #msg=msg.replace('data: ','')
        
        #match = re.search(r'data: (\{.*"message_id": ".*?"\})', msg, re.DOTALL)
        match = re.search(r'(\{"text":.*?"message_id": ".*?"\})', msg, re.DOTALL)
        # 如果找到匹配项，则提取并打印结果
        if match:
            msg = match.group(1)
            #print('返回结果：%s' % cmd)  # 打印请求体内容，看看实际接收到的数据
        else:
            return JsonResponse({'status': 'error', 'message': 'No match found'}, status=400)
        if not msg.strip():
            return JsonResponse({'status': 'error', 'message': 'Empty response'}, status=400)
        try:
            msg_new = json.loads(msg)
        except json.JSONDecodeError as e:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON msg: %s' % msg}, status=400)
        # msg_new = json.loads(msg)
        return JsonResponse({'message': msg_new['text'].replace('AI:','')})
        #return HttpResponse(request.body)
    else:
        # 如果不是POST请求，渲染并返回HTML页面
        return render(request, 'chat.html')

def if_show_command(messages,keyword_name,model,temperature):
    keyword_name = keyword_name
    user_message = messages
    model = model
    temperature = temperature

    prefix = '''
    # 角色：资深Robot Framework关键字开发工程师
    '''
    
    PREFIX=prefix
    question='不相关的内容'
    answer='''
    {
    "",
    }
    '''
    examples=[]
    examples.append({"question": question,"answer": answer,})

    example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIModel(model_name='m3e-base'),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # This is the number of examples to produce.
    k=5
    )

    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{answer}"),
    ]
    )
    chat_prompt_template_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    input_variables=["input"],
    )

    final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PREFIX),
        chat_prompt_template_prompt,
        ("human", "{input}"),
    ]
    )

    chat = JuLLMChatChat(temperature=temperature, model=model)
    chain = LLMChain(llm=chat, prompt=final_prompt, verbose=False)
    delimiter = "####"
    obj=KeywordDoc.objects.filter(keyword_name=keyword_name)
    keyword_doc=obj[0].keyword_doc_html
    keyword_example=obj[0].keyword_example
    system_message ="""
    
    你将提供服务查询。
    服务查询将使用"####"字符分隔。

    请用JSON的格式输出结果，不要添加其他无关的内容：
    {
    "get_info": <必须在下面的说明文档中找到get_info参数列表里面的参数值>
    "example":  <必须在下面的"关键字示例文档"或"关键字说明文档"中找到"示例"后，用根据用户输入的信息找到的get_info列表里对应的参数值替换示例中get_info的参数值'get_info=参数值',显示完整的"示例"内容，其他部分不变，只改变get_info的参数值>
        }
        
    请仔细阅读说明文档中get_info参数列表第一列的参数名称和第二列的参数说明
    如果用户提及了参数说明中的内容，则必须将其与参数列表中的正确参数值相关联。
    如果根据用户输入的信息未找到get_info列表里对应的参数值，则列出参数列表里所有可用的参数，提示用户没有匹配到可用的参数值。
    如果根据用户输入的信息找到get_info列表里对应的参数值,根据关键字示例中给出的示例格式，将get_info的参数值替换后按照示例的格式输出到Python 对象列表里作为第二个对象
    
    请仔细阅读下面HTML格式的"关键字说明文档"，保证理解了里面get_info参数的可用参数值和参数说明：
    '''
    %s
    '''
    请仔细阅读下面HTML格式的"关键字示例文档"，找出关键字名称所在行的内容，保证理解了里面关键字和参数的使用方法：
    '''
    %s
    '''
    
    注意：
    1.仅输出JSON格式的结果，里面的内容都使用双引号""，除了"get_info"对应的参数值和"example"对应的关键字示例，不要添加其他无关的内容。
    2.如果根据用户输入的信息在关键字说明文档中找到get_info列表里对应的参数值,找出关键字示例中包含关键字名称："%s"所在行，根据给出的示例格式，将get_info的参数值替换后按照示例的格式输出到Python 对象列表里作为第二个对象，显示完整的示例内容，其他部分不变，只改变get_info的参数值
    3.get_info=<必须在你已经仔细阅读过的关键字说明文档中找到get_info参数列表里面的参数值>
    4."example":  <必须在你已经仔细阅读过的的关键字示例文档中找到示例后，用根据用户输入的信息找到的get_info列表里对应的参数值替换示例中get_info的参数值'get_info=参数值',显示完整的示例内容，其他部分不变，只改变get_info的参数值>
    5.确保优先使用关键字示例文档里面能够找出的关键字示例，如果关键字示例里面有关键字"%s"的示例，就使用关键字示例里面的关键字示例格式，就不要用默认的关键字示例格式
    6.如果没有在关键字示例文档里面找到关键字示例，就使用关键字说明文档里面关键字示例中的关键字示例格式，就不要用默认的关键字示例格式
    7.如果在关键字示例文档和关键字说明文档里面都没有找到关键字示例格式，再用默认的关键字示例格式："${output}    %s    device=${dut1}    get_info=<必须在你已经仔细阅读过的关键字说明文档中找到get_info参数列表里面的参数值>"
    """ % (keyword_doc,keyword_example,keyword_name,keyword_name,keyword_name)

    few_shot_user_1 = """获取MinTxInt"""
    few_shot_assistant_1 = """ 
    {"get_info": "min_tx_int", \
     "example": "${min_tx_int1}    获取BFD邻居Local-disc信息    device=DUT1    local_disc=${local_disc}    get_info=min_tx_int"
    }
    """
#     few_shot_user_2 = """获取Link-Address"""
#     few_shot_assistant_2 = """ 
#     [{'get_info': 'link_address', \
# 'example': ${link_address_dut1}    获取ND信息    device=${dut1}    get_info=link_address}]
#     """
    
    
    messages =  [  
    {'role':'system', 
    'content': system_message}, 
    {'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},  
    {'role':'assistant', 'content': few_shot_assistant_1 },
    {'role':'user', 
    'content': f"{delimiter}{user_message}{delimiter}"},  
    
    ]

    # messages =  [  
    # {'role':'system', 
    # 'content': system_message}, 
    # {'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},  
    # {'role':'assistant', 'content': few_shot_assistant_1 },
    # {'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"},  
    # {'role':'assistant', 'content': few_shot_assistant_2 },
    # {'role':'user', 
    # 'content': f"{delimiter}{user_message}{delimiter}"},  
    
    # ]
    
    cmd='%s' % (messages)
    
    #cmd='%s\n%s' % (question['chatLog'],question['message'])
    print('输入内容：%s' % cmd)
    msg=chain.run(cmd)
    print('返回结果：%s' % msg)  # 打印请求体内容，看看实际接收到的数据
    #msg=msg.replace('data: ','')
    
    #match = re.search(r'data: (\{.*"message_id": ".*?"\})', msg, re.DOTALL)
    match = re.search(r'(\{"text":.*?"message_id": ".*?"\})', msg, re.DOTALL)
    # 如果找到匹配项，则提取并打印结果
    if match:
        msg = match.group(1)
        #print('返回结果：%s' % cmd)  # 打印请求体内容，看看实际接收到的数据
    else:
        return JsonResponse({'status': 'error', 'message': 'No match found'}, status=400)
    if not msg.strip():
        return JsonResponse({'status': 'error', 'message': 'Empty response'}, status=400)
    try:
        msg_new = json.loads(msg)
    except json.JSONDecodeError as e:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON msg: %s' % msg}, status=400)
    # msg_new = json.loads(msg)
    #return msg_new['text'].replace('AI:','')
    match1 = re.search(r'"example":\s*"([^"]+)"', msg_new['text'].replace('AI:',''))
    match2 = re.search(r"""'example':\s*'([^"]+)'""", msg_new['text'].replace('AI:',''))
    if match1:
        answer=match1.group(1)
    elif match2:
        answer=match2.group(1)
    else:
        answer='没有返回结果'
    return answer
    #return HttpResponse(request.body)

def split_show_command(messages,model,temperature):
    user_message = messages
    model = model
    temperature = temperature

    prefix = '''
    # 角色：资深Robot Framework关键字开发工程师
    '''
    
    PREFIX=prefix
    question='不相关的内容'
    answer='''
    {
    "",
    }
    '''
    examples=[]
    examples.append({"question": question,"answer": answer,})

    example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIModel(model_name='m3e-base'),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # This is the number of examples to produce.
    k=5
    )

    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{answer}"),
    ]
    )
    chat_prompt_template_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    input_variables=["input"],
    )

    final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PREFIX),
        chat_prompt_template_prompt,
        ("human", "{input}"),
    ]
    )

    chat = JuLLMChatChat(temperature=temperature, model=model)
    chain = LLMChain(llm=chat, prompt=final_prompt, verbose=False)
    delimiter = "####"
    # obj=KeywordDoc.objects.filter(keyword_name=keyword_name)
    # keyword_doc=obj[0].keyword_doc_html
    # keyword_example=obj[0].keyword_example
    system_message ="""
    
    你将根据用户输入信息提供"show命令行"内容提取和"show命令行"使用意图提取。
    用户输入信息将使用"####"字符分隔。

    请用JSON的格式输出结果，不要添加其他无关的内容：
    {
    "show_command": <根据用户输入信息提取出"show命令行"内容，必须在下面CVS格式的"show命令行"列表里面找到相同的"show命令行"内容，如果没有则输出空值>
    "to_do": <根据用户输入信息中不包含"show命令行"的部分总结出用户的使用意图，输出结果必须包含用户输入信息中所有英文字符的内容，不要添加无关内容>
        }
    
    请仔细阅读下面的"show命令行"列表，保证JSON的格式输出结果中"show_command"的内容必须是"show命令行"列表已有的内容：
    '''
    show命令行列表：
    show forwarding packet-drop counter detail
    show forwarding packet-drop counter interface
    show troubleshooting all
    show bfd neighbors srv6-policy-sl detail
    show bfd neighbors rsvp lsp brief
    show bfd location-board
    show bfd neighbors all detail
    show ipv6 pim mroute summary
    show ip pim mroute summary
    show bgp l2vpn martini
    show bgp l2vpn martini neighbor out
    show pwe3 signal fec129
    show pwe3 signal fec129 detail
    show pwe3 signal fec128 detail
    show running-config multicast6
    show running-config lspm
    show scc detect-event port-flowtype
    show scc detect-event cpu-flowtype
    show scc attack-trace brief
    show isis ipv6 route
    show isis ipv6 route summary
    show isis database summary
    show isis database level-1
    show isis adjacency summary
    show ipv6 ospf retransmission-list
    show ipv6 ospf database neighbor
    show ipv6 ospf database
    show ip ospf database neighbor
    show bgp l2vpn vpls summary
    show bgp l2vpn vpls route-statistics
    show bgp l2vpn vpls flap summary
    show bgp ipv6 unicast neighbor out
    show bgp ipv6 unicast neighbor in
    show bgp ipv4 unicast flap-statistics
    show bgp ipv4 unicast dampening-parameters
    show bgp ipv4 unicast dampened-paths
    show running-config hqos
    show cps gtsm statistics
    show cps flowtype send-quota-limit
    show cps flow limit interface
    show cps ctm-rate destcpu
    show cps ctm-queue statistics destcpu
    show dni-linked-path
    show evpn leaf-location
    show clockstatus
    show tcp6 brief
    show hostname
    show card-state brief
    show ip forwarding route
    show l3vpn-statistics status
    show l3vpn-statistics perfvalue
    show l2vpn forwardinfo vpws-vxlan
    show l2vpn-statistics perfvalue
    show l2vpn forwardinfo kompella
    show sr tunnels pce-auto-init brief
    show sr tunnels bgp-auto brief
    show sr static summary
    show sr static brief
    show sr resource static-tunnel perf-switch
    show mpls-tp oam meg meg-id
    show sr tunnels summary
    show sr tunnels hot-standby
    show segment-routing ti-lfa
    show sr policy forwarding-table segment-list
    show segment-routing microloop-prevention
    show mpls ldp bindings
    show mff interface
    show mff global-gateway
    show mff gateway
    show mff configure
    show mffv6 interface
    show mffv6 gateway
    show mffv6 configure
    show system-healthy hardware backplane-fabric-serdes
    show system-healthy hardware backplane-fabric-serdes
    show ip dhcp client user interface
    show ip dhcp pool
    show ip dhcp policy
    show ip dhcp relay server group
    show ip dhcp relay user total-count
    show ip dhcp relay user summary
    show ip dhcp relay user
    show ip dhcp server user total-count
    show ip dhcp server user summary
    show ip dhcp server user
    show ipv6 dhcp client user interface
    show ipv6 dhcp relay user
    show ipv6 dhcp server user
    show monitor session
    show ip pim bsr
    show ip pim interface brief
    show ip pim interface
    show ip pim mroute
    show ip pim neighbor brief
    show ip pim nexthop
    show ip pim rp hash
    show ip pim snooping neighbor-info
    show ip pim snooping port-info vlan
    show ipv6 pim interface brief
    show ipv6 pim interface
    show ipv6 pim mroute
    show ipv6 pim neighbor brief
    show ip pim nexthop
    show ipv6 pim rp hash
    show ipv6 pim rp mapping static
    show ipv6 pim rp mapping
    show pimv6 snooping neighbor-info
    show pimv6 snooping port-info vlan
    show ipv4-access-groups
    show ipv4-access-lists
    show ipv4-acl-log access-list
    show ipv4-acl-log interface cgei-0
    show ipv4-mixed-access-groups
    show ipv4-mixed-access-lists
    show ipv4-mixed-acl-log access-list
    show ipv4-mixed-acl-log interface cgei-0
    show ipv4-vxlan-access-groups
    show ipv4-vxlan-access-lists
    show ipv4-vxlan-acl-log access-list
    show ipv6-access-groups
    show ipv6-access-lists
    show ipv6-acl-log access-list
    show ipv6-acl-log interface cgei-0
    show ipv6-mixed-access-groups
    show ipv6-mixed-access-lists
    show ipv6-mixed-acl-log access-list
    show ipv6-mixed-acl-log interface cgei-0
    show ipv6-vxlan-access-groups
    show ipv6-vxlan-access-lists
    show ipv6-vxlan-acl-log access-list
    show link-access-groups
    show link-access-lists
    show link-acl-log access-list
    show link-acl-log interface cgei-0
    show running-config icbg
    show ip mdt
    show ip mvpn instance
    show packet-capture status
    show packet-capture packet
    show ip mroute summary
    show ip mroute
    show ip msdp count
    show ip msdp peer
    show ip msdp sa-cache
    show ip msdp summary
    show ip pim rp
    show ip pim rp hash
    show ip pim rp mapping
    show ipv6 mroute summary
    show ipv6 mroute
    show ip mroute
    show ipv6 pim bsr
    show ip igmp groups
    show ip igmp groups
    show ip igmp interface
    show lldp config interface
    show lldp config
    show lldp entry
    show lldp statistic
    show lldp neighbor brief
    show lldp neighbor
    show ntp associations
    show ntp status
    show running-config ntp
    show ssm current-clock
    show running-config igmp-snoop
    show ip igmp snooping group
    show ip igmp snooping mr-port-info
    show ip igmp snooping port-info evpn-vpls
    show ip igmp snooping port-info vlan
    show ip igmp snooping query
    show ip igmp snooping statistic packet vlan
    show ip igmp snooping summary mr-port-info
    show ip igmp snooping summary port-info
    show ip igmp snooping summary port-info vlan
    show ip igmp snooping summary vlan
    show ip igmp snooping summary
    show ip igmp snooping vlan
    show ip igmp snooping
    show vsc config
    show vsc information
    show vsc link
    show vsc next
    show vsc pfu-port-status
    show gvrp statistics interface
    show sflow agent address
    show sflow cp
    show sflow fs
    show sflow interface
    show sflow receiver
    show clients port
    show logfile
    show vxlan tunnel
    show vrrp interface
    show vrrp ipv4 brief
    show vrrp ipv6 brief
    show segment-routing ipv4-mpls adjacency-sid
    show segment-routing ipv6-mpls adjacency-sid
    show sr policy brief
    show sr policy summary
    show sr policy traffic-statistics
    show sr policy
    show segment-routing srv6 local-sid
    show segment-routing srv6 locator
    show sr tunnels brief
    show sr tunnels te
    show running-config dns
    show ipv6 addr-pool configure
    show ipv6 addr-pool exclude
    show ipv6 addr-pool usage-peak
    show running-config ippoolv6
    show sibtp
    show class-map
    show diffserv domain
    show diffserv-domain default
    show policy-map
    show qos diffserv-domain
    show qos priority-remark
    show qos priority
    show qos-buffer queue interface
    show qos-buffer sys
    show qos-queue-statistics interface
    show traffic-policy interface
    show traffic-policy-statistics interface
    show intf-statistics threshold
    show intf-statistics utilization detail
    show intf-statistics utilization
    show ipv6 mld groups
    show ipv6 mld groups summary
    show ipv6 mld interface
    show ip vrf brief
    show ip vrf detail
    show ip vrf summary
    show virtual-network-device
    show supervlan-pool
    show supervlan-v6pool
    show supervlan-v6
    show supervlan
    show spantree interface
    show spantree statistics
    show spantree tc-bpdu statistics
    show clock
    show port-group
    show nd l2-proxy-vpls
    show nd
    show nd6 cache
    show clock
    show running-config environment-config all
    show running-config environment-config
    show fan extend
    show forwarding packet-drop counter
    show fan
    show forwarding packet-drop counter interface
    show forwarding packet-drop information
    show forwarding-resource-detail
    show forwarding-resource-hardware all
    show opticalinfo brief
    show opticalinfo
    show port-defend statistics
    show forwarding-resource-detail
    show port-defend
    show power board
    show power extend
    show power fan
    show power run-status-alarm
    show power
    show resource statistics fib
    show resource statistics mac
    show serial number
    show tcp brief
    show temperature environment-threshold
    show temperature detail
    show voltage
    show bfd neighbors ip brief
    show bfd neighbors l2 brief
    show bfd neighbors ldp detail
    show bfd neighbors ldp brief
    show bfd neighbors ip detail
    show bgp all summary
    show bgp ipv4 flowspec detail
    show bgp ipv4 flowspec in detail
    show bgp ipv4 flowspec route-statistics
    show bgp ipv4 flowspec summary
    show bgp ipv6 flowspec
    show bgp ipv6 unicast detail
    show bgp ipv6 unicast labels
    show bgp ipv6 unicast route-statistics
    show bgp ipv6 unicast summary
    show bgp ipv6 unicast
    show bgp l2vpn evpn flap-statistics
    show bgp l2vpn evpn ip-vrf
    show bgp l2vpn evpn labels
    show bgp l2vpn evpn route-statistics
    show bgp l2vpn evpn statistics
    show bgp l2vpn evpn summary
    show bgp l2vpn evpn
    show bgp bgp remote prefix-sid
    show bgp vpnv4 flowspec
    show bgp vpnv4 flowspec in detail
    show bgp vpnv4 flowspec summary
    show bgp vpnv4 unicast
    show bgp vpnv4 unicast route-statistics
    show bgp vpnv4 unicast summary
    show bgp vpnv6 flowspec
    show bgp vpnv6 flowspec in detail
    show bgp vpnv6 flowspec summary
    show bgp vpnv6 unicast
    show bgp vpnv6 unicast route-statistics
    show bgp all summary
    show running-config bgp
    show ip bgp labels
    show ip bgp route detail
    show ip bgp neighbor
    show ip forwarding route statistic
    show ip forwarding route summary
    show ip ospf interface
    show ip ospf neighbor detail
    show ip ospf virtual-links
    show ip ospf neighbor
    show ip rip database
    show ip rip networks
    show ip rip statistics
    show ip rip neighbors
    show ipv6 forwarding route statistic
    show ipv6 forwarding route summary
    show ipv6 ospf database
    show ipv6 ospf interface
    show ipv6 ospf neighbor
    show ipv6 ospf virtual-links
    show ipv6 ospf neighbor
    show running-config ospfv3
    show isis fast-reroute-topology lfa
    show isis adjacency process-id
    show running-config ospfv2
    show running-config ospfv3
    show gvrp statistics interface
    show lacp
    show lacp mc-lag-global
    show lacp sys-id
    show mc-lag consistency-check dhcp relay
    show counter inbound
    show counter outbound
    show udld
    show arp statistics
    show arp
    show arp l2-proxy-vpls
    show vlan private-map
    show vlan statistics vlan
    show vlan translate statistics session
    show vlan translation
    show vlan
    show cpu process statistics
    show cpuload-threshold
    show forwarding-resource
    show forwarding-resource all
    show forwarding-resource tunnel
    show pkt-drop-statistics
    show processor
    show eld detect-entity domain
    show eld detect-entity interface
    show eld detect-entity
    show eld loop-point domain
    show eld loop-point interface
    show eld loop-point
    show running-config eld
    show cfm mp
    show cfm resource-statistics
    show ztp device-list
    show ztp device
    show ztp
    show dcn config global
    show dcn config mngip
    show dcn config-ipv6 global
    show dcn config-ipv6 mngip
    show dcn status l2
    show dcn status l3
    show dcn status-ipv6 l2
    show dcn status-ipv6 l3
    show ssh
    show running-config ssh
    show sqa-result dns
    show sqa-result ftp
    show sqa-result generalflow
    show sqa-result http
    show sqa-result icmpjitter
    show sqa-result icmp
    show sqa-result light-twamp complete
    show sqa-result snmp
    show sqa-result tcp
    show sqa-result udpjitter
    show sqa-result udp
    show sqa-server tcp
    show sqa-test
    show ipv6 forwarding all-routes
    show ipv6 forwarding backup route
    show ipv6 forwarding route
    show ipv6 protocol routing
    show tunnel-policy instance-info
    show tunnel-policy selecting-result
    show bier forwarding-table bit-position
    show bier forwarding-table bitstring-length
    show bier forwarding-table fec
    show bier forwarding-table g-bier-bift-id
    show bier forwarding-table bier-nexthop
    show bier forwarding-table non-mpls-bift-id
    show ip rsvp authentication
    show ip rsvp hello bfd
    show ip rsvp hello graceful-restart
    show ip rsvp hello instance detail
    show ip rsvp hello instance summary
    show ip rsvp refresh parameter
    show ip rsvp refresh reduction
    show zdp neighbour interface
    show zdp neighbour mac
    show zdp neighbour
    show running-config load-balance-enhance all
    show tunnel-group
    show ptp output-clock-info
    show ptp grandmaster-clock-info
    show ptp time-source-clock-info interface gps-1
    show ptp time-offset
    show ptp status
    show ptp port erbest interface gei-1
    show ptp port-state xgei-1
    show ptp steps-from-grandmaster
    show ptp mean-link-delay
    show route-map
    show aps erps vlan-group
    show interface brief
    show running-config-interface all
    show running-config-interface
    show interface description
    show interface
    show ip interface brief
    show ip interface
    show ipv6 interface brief
    show ipv6 interface
    show performance threshold
    show history
    show ip local pool configure
    show ip local pool statistics total
    show ip local pool
    show te-ecmp-group all
    show garp config
    show simulate-hash L2 dmac
    show simulate-hash l3 ipv4 dip
    show simulate-hash l3 ipv6 dip
    show running-config ufp-config
    show nd-snooping bind-table
    show nd-snooping pfefix
    show alarm current
    show alarm history
    show alarm notification
    show logging alarm
    show ipv6 prefix-list summary
    show ipv6 prefix-list
    show lsp-group all
    show mpls forwarding-table
    show mpls label manage-mode
    show mpls label manage
    show mpls label usage
    show mpls ldp neighbor detail instance id
    show mpls oam information local
    show mpls oam information remote
    show mpls oam statistics local
    show mpls oam statistics remote
    show mpls traffic-eng fast-reroute bw-protect
    show mpls traffic-eng fast-reroute
    show mpls traffic-eng interface brief
    show mpls traffic-eng interface detail
    show mpls traffic-eng local-tunnels
    show mpls traffic-eng mtunnels summary
    show mpls traffic-eng static autoroute
    show mpls traffic-eng static brief
    show mpls traffic-eng static summary
    show mpls traffic-eng static
    show mpls traffic-eng tunnels backup
    show mpls traffic-eng tunnels brief
    show mpls traffic-eng tunnels interface
    show mpls traffic-eng tunnels local-tunnel brief
    show mpls traffic-eng tunnels remote-tunnel
    show mpls traffic-eng tunnels summary
    show mpls traffic-eng tunnels
    show mpls-tp oam meg
    show rsvp bandwidth interface
    show ip dhcp snooping configure
    show samgr bind track-group
    show samgr bind track
    show samgr brief
    show samgr track-group
    show samgr track
    show running-config of all
    show running-config of
    show openflow ofls
    show anti-dos-statistics
    show anti-dos-statistics abnormal shelf
    show anti-dos-statistics ping-flood shelf
    show anti-dos
    show show l2vpn ac-using
    show l2vpn brief
    show l2vpn evpn broadcast list
    show l2vpn forwardinfo
    show l2vpn forwardinfo evpn-vpws
    show show l2vpn instance-name
    show l2vpn protectgroup
    show show l2vpn pw-name pw
    show show l2vpn pw-using
    show l2vpn brief
    show bfd neighbors all brief
    show bfd neighbors lag brief
    show bfd neighbors sr-be brief
    show bfd neighbors sr-be detail
    show bfd neighbors sr-mpls-policy-sl brief
    show bfd neighbors sr-mpls-policy brief
    show bfd neighbors srv6-be brief
    show bfd neighbors srv6-be detail
    show bfd neighbors srv6-evpn-vpws brief
    show bfd neighbors srv6-policy-sl brief
    show bfd neighbors srv6-policy brief
    show bfd neighbors te tunnel brief
    show bfd neighbors te tunnel detail
    show running-config bfd
    show backplane
    show patch effective
    show patch
    show running-config system-config
    show synchronization
    show le
    show version
    show boot-firmware
    show hardware
    show board-info
    show running-config switchvlan all
    show base-mac
    show basemac
    show lacp sys-id
    show base-mac
    show basemac
    show mac bridge-domain summary
    show mac l2vpn
    show mac limit-maximum
    show mac move-dampening vlan
    show mac move-dampening vpls
    show mac table summary
    show mac vpls summary
    show mac table
    show running-config mac
    show mac vpls instance
    show ip forwarding all-routes
    show ip forwarding route
    show ip protocol routing summary
    show ip protocol routing
    show ipv6 protocol routing summary
    show mldsnoop evpn-vpls
    show mldsnoop mrport-info
    show mldsnoop port-info evpn-vpls
    show mldsnoop port-info ring-tree
    show mldsnoop port-info vlan
    show mldsnoop port-info vpls
    show mldsnoop port-info vrf
    show mldsnoop summary
    show mldsnoop vlan
    show mldsnoop vpls
    show mldsnoop
    show twamp client connection
    show twamp client session
    show twamp result packet-latency
    show twamp result packet-lost
    show twamp server connection
    show twamp server session
    show ip dhcp snooping database
    show ipv6 dhcp snooping database
    show evpn df-election
    show evpn esi-location
    show ipv6 prefix-pool configure
    show ipv6 prefix-pool exclude
    show ipv6 prefix-pool usage-peak
    show running-config ippoolv6
    show ipv6 pmtu
    show running-config ipv6 all
    show running-config ip
    show running-config ip all
    show cps attack-sample current top
    show cps attack-sample history top
    show cps attack-source
    show cps cpu-cir queue rate-limit
    show cps cpu-cir statistics
    show cps flow statistics flowtype
    show cps flow statistics interface
    show cps flow statistics
    show cps packet-drop statistics
    show cps protocol-to-queue
    show aps erps vlan-group
    show aps linear-protect
    show l2vpn summary
    show device ipv4-access-groups
    show device ipv4-mixed-access-groups
    show ethernet-oam
    show arp proxy statistics
    show synchronization detail
    show load-mode
    show boot-firmware
    show vsc reserve-port config
    show vsc reserve-port next
    show inband oam global
    show inband oam alternate-marking flow-id
    show inband oam alternate-marking flow-name
    show inband oam statistics-result flow-id
    show inband oam alternate-marking flow-name
    show ip forwarding route
    show ip forwarding route summary all
    show ipv6 forwarding route summary all
    show ip forwarding route summary global
    show ipv6 forwarding route summary global
    show fip-snooping configure
    show fip-snooping enode
    show fip-snooping fcf
    show fip-snooping session
    show fip-snooping vlan
    show schp-info groupid
    show cps flow statistics
    show cps flow statistics
    show running-config udld
    show configuration trial status
    show configuration prime-key
    show configuration commit list
    show running-config ipv6-tunnel
    show ip pim traffic
    show ipv6 pim traffic
    show flexe
    show flexe-subclient resource num
    show alarm current brief
    show ip prefix-list summary
    show ip prefix-list
    show bgp vpnv4 unicast
    show bgp vpnv6 unicast
    show bgp l2vpn evpn mac detail
    show bgp ipv4 unicast detail
    show isis ipv4 route summary
    show isis ipv4 route
    show ip bgp route
    show bgp ipv4 unicast route-statistics
    show running-config rip
    show bgp evpn mac-vrf vpls
    show ipv6 rip database
    show bgp evpn mac
    show mpls forwarding-table summary
    show mpls ldp instance statistic
    show mpls ldp iccp
    show mpls ldp igp sync brief instance
    show mpls ldp igp sync detail instance
    show mpls ldp neighbor brief instance
    show mpls ldp bindings summary
    show linkage-group
    show nd proxy statistics
    show control-protocol priority
    show forwarding packet-drop counter detail np-mc
    show forwarding packet-drop counter interface
    show qos-statistics interface
    show qos-buffer shelf
    show ipv6 dhcp snooping configure
    show ipv6 dhcp snooping vlan
    show ipv6 dhcp snooping trust
    show keychip exception
    show keychip policy
    show peb access-pe entry
    show license emergency
    show telemetry destination
    show telemetry sensor-path brief
    show bfd statistics
    show troubleshooting module bfd
    show bfd neighbors
    show ip policy-statistics
    show running-config-interface all
    show running-config interface-performance all
    show segment-routing srv6-usid locator
    show vxlan vni summary
    show vxlan peer
    show ip mvpn ad-route summary
    show version
    show spantree instance
    show hostname
    show forwarding-resource-mode
    show l2vpn ethernet-segments forwarding detail
    '''
    
    注意：
    1.仅输出JSON格式的结果，里面的内容都使用双引号""，除了"show_command"对应的"show命令行"，"to_do"对应的用户使用意图，不要添加其他无关的内容。
    2."show_command": <根据用户输入信息提取出"show命令行"内容，必须在上面的"show命令行"列表里面找到相同的"show命令行"内容，如果没有则输出空值>
    3."to_do": <根据用户输入信息中不包含"show命令行"的部分总结出用户的使用意图，输出结果必须包含用户输入信息中所有英文字符的内容，不要添加无关内容>

    """

    # few_shot_user_1 = """show interface获取MPLS utilization output信息"""
    # few_shot_assistant_1 = """ 
    # {"show_command": "show interface", \
    #  "keyword_name": "获取Interface信息",\
    #  "to_do": "获取MPLS utilization output信息"
    # }
    # """
#     few_shot_user_2 = """获取Link-Address"""
#     few_shot_assistant_2 = """ 
#     [{'get_info': 'link_address', \
# 'example': ${link_address_dut1}    获取ND信息    device=${dut1}    get_info=link_address}]
#     """
    
    
    messages =  [  
    {'role':'system', 
    'content': system_message}, 
    {'role':'user', 
    'content': f"{delimiter}{user_message}{delimiter}"},  
    
    ]

    # messages =  [  
    # {'role':'system', 
    # 'content': system_message}, 
    # {'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},  
    # {'role':'assistant', 'content': few_shot_assistant_1 },
    # {'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"},  
    # {'role':'assistant', 'content': few_shot_assistant_2 },
    # {'role':'user', 
    # 'content': f"{delimiter}{user_message}{delimiter}"},  
    
    # ]
    
    cmd='%s' % (messages)
    
    #cmd='%s\n%s' % (question['chatLog'],question['message'])
    print('输入内容：%s' % cmd)
    msg=chain.run(cmd)
    print('返回结果：%s' % msg)  # 打印请求体内容，看看实际接收到的数据
    #msg=msg.replace('data: ','')
    
    #match = re.search(r'data: (\{.*"message_id": ".*?"\})', msg, re.DOTALL)
    match = re.search(r'(\{"text":.*?"message_id": ".*?"\})', msg, re.DOTALL)
    # 如果找到匹配项，则提取并打印结果
    if match:
        msg = match.group(1)
        #print('返回结果：%s' % cmd)  # 打印请求体内容，看看实际接收到的数据
    else:
        return JsonResponse({'status': 'error', 'message': 'No match found'}, status=400)
    if not msg.strip():
        return JsonResponse({'status': 'error', 'message': 'Empty response'}, status=400)
    try:
        msg_new = json.loads(msg)
    except json.JSONDecodeError as e:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON msg: %s' % msg}, status=400)
    # msg_new = json.loads(msg)
    #return msg_new['text'].replace('AI:','')
    match1 = re.search(r'"show_command":\s*"([^"]+)"', msg_new['text'].replace('AI:',''))
    match2 = re.search(r"""'show_command':\s*'([^"]+)'""", msg_new['text'].replace('AI:',''))
    if match1:
        answer1=match1.group(1)
    elif match2:
        answer1=match2.group(1)
    else:
        answer1='show命令行没有返回结果'
    match1 = re.search(r'"to_do":\s*"([^"]+)"', msg_new['text'].replace('AI:',''))
    match2 = re.search(r"""'to_do':\s*'([^"]+)'""", msg_new['text'].replace('AI:',''))
    if match1:
        answer2=match1.group(1)
    elif match2:
        answer2=match2.group(1)
    else:
        answer2='用户意图没有返回结果'
    return answer1, answer2
    #return HttpResponse(request.body)
    
