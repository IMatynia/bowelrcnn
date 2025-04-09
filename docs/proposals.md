Koncepcje architektury do problemu wykrywania dźwięków jelitowych

1. Modyfikacja orginalnej sieci CRNN:
    - architektura pozostaje w większości bez zmian
    - trenowanie sieci aby była w stanie wykrywać początki oraz końce dźwięku, zamiast przynależność wycinków
    - wykrywanie końców odbywałoby się na podstawie plików niezmodyfikowanych czasowo, natomiast początki wykrywane by były na podstawie dźwięków odwróconych w osi czasu. Pozwoliłoby to na przekazanie do rekurencyjnej części sieci wystarczającego kontekstu do podjęcia decyzji. (Zakładamy, że wykrycie początku dźwięku jelitowego bez żadnego kontekstu jest trudniejszym zadaniem niż wykrycie końca)
    - inferencja operowałaby na podstawie obszarów wyznaczonych pomiędzy początkami a końcami. Wybierane by były lokalnie najbardziej prawdopodobne początki i końce

2. Użycie konwolucyjnych sieci wykrywających wzorce w obrazie spektrogramu.
    1. Preprocesowanie wejścia:
        > Surowe wejście jest postaci plików audio
        - Nałożenie filtru 
            - Ograniczenie zakresu częstotliwości do obszarów zainteresowania: od 50Hz do 3kHz (data_exploration.ipynb)
        - Wygenerowanie obrazu spektrogramu MEL
        - normalizacja
    2. Wejście sieci:
        - Spektrogram obszaru audio w postaci obrazu RGB o zadanej rozdzielczości. Wielkość obszaru jest większa niż najdłuższy możliwy rozważany dźwięk jelitowy. Z przeglądu danych wynika że stanowcza większość dźwięków jelitowych jest długości od 0.01s do 0.1s, średnio około 0.03. Rozważanym obszarem może w takim wypadku być obszar dwukrotnie większy, 0.2s (data_exploration.ipynb)
    3. Wyjście sieci:
        - Detekcja - czy w danym rozważanym obszarze znajduje się dźwięk jelitowy? [0..1] pewność czy wystaje
        - Clipping - czy dźwięk wystaje poza obszar? [0..1], pewność czy wystaje
        - Pokrycie dźwięku - jakie jest spodziewane pokrycie dźwięku? [0..1]
        - Parametry bounding-box (przesunięcie środka [0..1] (0.5 oznacza brak przesunięcia) oraz skala [0..1])
            - dźwięk długości 0.01 znajdujący się na środku okna 0.2 oznaczony by był wartością (0.5, 0.05) 
    4. Architektura sieci: (przykładowa, sprawdzane by było kilka konfiguracji)
        - wejście, macierz HxWx3
        - Conv 7x7x3
        - Batch norm
        - Max pooling
        - Conv 3x3
        - Batch norm
        - Conv 3x3
        - Batch norm
        - Max pooling
        - Fully connected 256 (dropout)
        - ReLu
        - Batch norm
        - Fully connected 256 (dropout)
        - wyjście, aktywacja Relu, fully connected layer, 5 wartości.
    5. Inferencja:
        - odrzucamy proponowane dźwięki jelitowe dla których 'detekcja' jest niska
        - obszary, które wychodzą poza okno z wysoką wartością 'clipping' mogą zwiększać pewność obszarów które pokrywają
        - wykluczone są obszary z małym 'pokryciem'
        - boundingbox wyznacza obszar występowania dźwięku jelitowego (jeśli nie został odrzucony)
    6. Augmentacja danych:
        - w celu zmniejszenia szansy na przeuczenie wykorzystana będzie augmentacja danych
        - augmentacje które można rozważyć:
            - dodanie szumu losowego
            - zakrywanie pasm sygnału
            - modulacja amplitudy 
    7. Semi-supervised learning:
        - wypróbowane może być podejście semi-supervised learning dla nieadnotowanych danych
    6. Funkcja celu:
        - maksymalizowana będzie wartość IOU (Intersection over union) z wyznaczonych bounding-boxów
            - detekcja powinna być równa 1
            - clipping powinien być równy 0 lub 1 w zaleśności od tego czy dźwięk jest ucięty w obszarze
            - pokrycie zależne od clippingu
        - w przypadku gdy w badanym obszarze nie ma dźwięku jelitowego:
            - detekcja powinna być równa zero
            - clipping powinnien być równy zero
            - pokrycie powinno być równe zero
            - bounding box powinien wskazywać na środek obszaru (0.5 offset), oraz skalę 0
            - wszystkie te wartości zostaną uśrednonie z wagą by uzyskać końcową wartość straty.
        - zbiór jest niezbilansowany (ok. 95% nagrania nie zawiera dźwięków jelitowych) - będzie to brane pod uwagę podczas treningu