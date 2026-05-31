from pydantic import BaseModel, Field
from typing import List

DISTRICTS = [
    'Ajeroport', 'Akademicheskoe', 'Alekseevskoe', "Altuf'evskoe", 'Arbat',
    'Babushkinskoe', 'Basmannoe', 'Begovoe', 'Beskudnikovskoe', 'Bibirevo',
    'Birjulevo Vostochnoe', 'Birjulevo Zapadnoe', 'Bogorodskoe', 'Brateevo',
    'Butyrskoe', 'Caricyno', 'Cheremushki', "Chertanovo Central'noe",
    'Chertanovo Juzhnoe', 'Chertanovo Severnoe', 'Danilovskoe', 'Dmitrovskoe',
    'Donskoe', 'Dorogomilovo', 'Filevskij Park', 'Fili Davydkovo',
    'Gagarinskoe', "Gol'janovo", 'Golovinskoe', 'Hamovniki',
    'Horoshevo-Mnevniki', 'Horoshevskoe', 'Hovrino', 'Ivanovskoe',
    'Izmajlovo', 'Jakimanka', 'Jaroslavskoe', 'Jasenevo', 'Juzhnoe Butovo',
    'Juzhnoe Medvedkovo', 'Juzhnoe Tushino', 'Juzhnoportovoe', 'Kapotnja',
    "Kon'kovo", 'Koptevo', 'Kosino-Uhtomskoe', 'Kotlovka', "Krasnosel'skoe",
    'Krjukovo', 'Krylatskoe', 'Kuncevo', 'Kurkino', "Kuz'minki", 'Lefortovo',
    'Levoberezhnoe', 'Lianozovo', 'Ljublino', 'Lomonosovskoe',
    'Losinoostrovskoe', "Mar'ina Roshha", "Mar'ino", 'Marfino', 'Matushkino',
    'Meshhanskoe', 'Metrogorodok', 'Mitino', 'Molzhaninovskoe',
    "Moskvorech'e-Saburovo", 'Mozhajskoe', 'Nagatino-Sadovniki',
    'Nagatinskij Zaton', 'Nagornoe', 'Nekrasovka', 'Nizhegorodskoe',
    'Novo-Peredelkino', 'Novogireevo', 'Novokosino', 'Obruchevskoe',
    'Ochakovo-Matveevskoe', 'Orehovo-Borisovo Juzhnoe',
    'Orehovo-Borisovo Severnoe', 'Ostankinskoe', 'Otradnoe', 'Pechatniki',
    'Perovo', 'Pokrovskoe Streshnevo', 'Poselenie Desjonovskoe',
    'Poselenie Filimonkovskoe', 'Poselenie Kievskij', 'Poselenie Klenovskoe',
    'Poselenie Kokoshkino', 'Poselenie Krasnopahorskoe',
    'Poselenie Marushkinskoe', 'Poselenie Mihajlovo-Jarcevskoe',
    'Poselenie Moskovskij', 'Poselenie Mosrentgen',
    'Poselenie Novofedorovskoe', 'Poselenie Pervomajskoe',
    'Poselenie Rjazanovskoe', 'Poselenie Rogovskoe', 'Poselenie Shhapovskoe',
    'Poselenie Shherbinka', 'Poselenie Sosenskoe', 'Poselenie Vnukovskoe',
    'Poselenie Voronovskoe', 'Poselenie Voskresenskoe', 'Preobrazhenskoe',
    'Presnenskoe', 'Prospekt Vernadskogo', 'Ramenki', 'Rjazanskij',
    'Rostokino', 'Savelki', 'Savelovskoe', 'Severnoe', 'Severnoe Butovo',
    'Severnoe Izmajlovo', 'Severnoe Medvedkovo', 'Severnoe Tushino',
    'Shhukino', 'Silino', 'Sokol', "Sokol'niki", 'Sokolinaja Gora',
    'Solncevo', 'Staroe Krjukovo', 'Strogino', 'Sviblovo', 'Taganskoe',
    "Tekstil'shhiki", 'Teplyj Stan', 'Timirjazevskoe', 'Troickij okrug',
    'Troparevo-Nikulino', 'Tverskoe', 'Veshnjaki', 'Vnukovo', 'Vojkovskoe',
    'Vostochnoe', 'Vostochnoe Degunino', 'Vostochnoe Izmajlovo',
    'Vyhino-Zhulebino', "Zamoskvorech'e", 'Zapadnoe Degunino', 'Zjablikovo',
    'Zjuzino',
]


class ApartmentFeatures(BaseModel):
    full_sq: float = Field(..., gt=0, le=1000, description="Общая площадь, м²")
    life_sq: float = Field(..., gt=0, le=1000, description="Жилая площадь, м²")
    floor: int = Field(..., ge=1, le=77, description="Этаж")
    max_floor: int = Field(..., ge=1, le=77, description="Этажность дома")
    build_year: int = Field(..., ge=1900, le=2030, description="Год постройки")
    num_room: int = Field(..., ge=1, le=19, description="Количество комнат")
    kitch_sq: float = Field(..., gt=0, le=200, description="Площадь кухни, м²")
    sub_area: str = Field(..., description="Район")


class PredictionResult(BaseModel):
    price: float
    price_formatted: str
    features: ApartmentFeatures


class BatchPredictionResult(BaseModel):
    predictions: List[dict]
    count: int