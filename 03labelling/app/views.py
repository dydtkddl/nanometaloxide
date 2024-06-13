from django.shortcuts import render, redirect
from .models import Title, LabeledTitle
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.http import require_GET
# 엑셀 파일 읽기 및 데이터베이스에 저장하는 로직 (생략 가능)
def load_data(request):
    file_path = request.GET.get("filename")
    df = pd.read_excel(file_path)
    count = 0
    for _, row in df.iterrows():
        # title이 중복되는지 확인
        if not Title.objects.filter(title=row['title']).exists():
            Title.objects.create(
                keyword=row['keyword'],
                title=row['title'],
                translated_title=row['Translated Title']
            )
        else: 
            count+=1
    print("duples : %s"%count)
    return redirect('label_titles')

# 라벨링 페이지
def label_titles(request):
    if request.method == 'POST':
        title_id = request.POST.get('title_id')
        oxides = request.POST.getlist('oxide')
        nanoparticle_status = request.POST.getlist('is_nanoparticle')
        title = Title.objects.get(id=title_id)

        for oxide, is_nanoparticle in zip(oxides, nanoparticle_status):
            is_nanoparticle = is_nanoparticle == 'yes'
            LabeledTitle.objects.create(title=title, oxide=oxide, is_nanoparticle=is_nanoparticle)

        return redirect('label_titles')

    title = Title.objects.exclude(labeledtitle__isnull=False).first()
    if not title:
        return render(request, 'app/completed.html')

    total_titles = Title.objects.count()
    completed_titles = LabeledTitle.objects.values('title_id').distinct().count()

    oxides = {
        "SiO2": ["Silicon Dioxide", "Silica", "Quartz", "Cristobalite", "Tridymite"],
        "ZnO": ["Zinc Oxide", "ZnO", "Zincite"],
        "TiO2": ["Titanium Dioxide", "Titania", "Rutile", "Anatase", "Brookite"],
        "Al2O3": ["Aluminum Oxide", "Alumina", "Corundum", "Alundum"],
        "CuO": ["Copper(II) Oxide", "CuO", "Tenorite"],
    }

    context = {
        'title': title,
        'oxides': oxides,
        'total_titles': total_titles,
        'completed_titles': completed_titles,
        'keywords': title.keyword
    }

    return render(request, 'app/label_titles.html', context)

@require_GET
def get_labeling_info(request):
    start_index = request.GET.get('s', 0)  # 시작 인덱스, 디폴트 0
    quantity = request.GET.get('q', None)  # 가져올 개수, 디폴트 None (모든 데이터)

    try:
        start_index = int(start_index)
        if quantity is not None:
            quantity = int(quantity)
    except ValueError:
        return JsonResponse({'error': 'Invalid query parameters'}, status=400)

    titles = Title.objects.all()

    if quantity is not None:
        titles = titles[start_index:start_index + quantity]
    else:
        titles = titles[start_index:]

    data = {
        'title': [],
        'is_nanoparticle': [],
        'main_subject': []
    }

    for title in titles:
        data['title'].append(title.title)
        
        labeled_titles = LabeledTitle.objects.filter(title=title)
        is_nanoparticle_row = [0, 0, 0, 0, 0, 0]  # Zinc Oxide, Silicon Dioxide, Titanium Dioxide, Aluminum Oxide, Copper(II) Oxide, Other
        main_subject_row = [0, 0, 0, 0, 0, 0]

        for labeled_title in labeled_titles:
            oxide = labeled_title.oxide
            is_nanoparticle = labeled_title.is_nanoparticle

            if oxide == 'ZnO':
                index = 0
            elif oxide == 'SiO2':
                index = 1
            elif oxide == 'TiO2':
                index = 2
            elif oxide == 'Al2O3':
                index = 3
            elif oxide == 'CuO':
                index = 4
            else:
                index = 5  # Other

            is_nanoparticle_row[index] = 1 if is_nanoparticle else 0
            main_subject_row[index] = 1  # Assuming every labeled title is a main subject
            
        data['is_nanoparticle'].append(is_nanoparticle_row)
        data['main_subject'].append(main_subject_row)

    return JsonResponse(data)