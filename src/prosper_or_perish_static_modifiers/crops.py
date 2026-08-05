from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


WATER_MODES = ("rainfed", "irrigated")
# GAEZ YXX/YLX and MapSPAM yields are native per hectare; publish as per km².
HECTARES_PER_KM2 = 100.0
GAEZ_V5_BASE = (
    "https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET/RES05-YXX"
)

# id, suffix, label, unit, UI group, zero_is_missing, compatible water_modes
METRICS = (
    (
        "production_density",
        "production_density_kg_dm_total_km2",
        "Production density",
        "kg DM / km²",
        "Potential",
        True,
        WATER_MODES,
    ),
    (
        "yield",
        "yield_kg_dm_suitable_km2",
        "Yield",
        "kg DM / suitable km²",
        "Potential",
        True,
        WATER_MODES,
    ),
    ("suitable_fraction", "suitable_fraction", "Suitable fraction", "fraction", "Potential", True, WATER_MODES),
    ("suitability_index", "suitability_index", "Suitability class", "class 1 best → 9 worst", "Potential", True, WATER_MODES),
    ("irrigation_need", "net_irrigation_requirement_mm", "Irrigation need", "mm", "Water & calendar", False, ("irrigated",)),
    ("cycle_start", "crop_cycle_start_doy", "Cycle start", "day of year", "Water & calendar", True, ("irrigated",)),
    ("cycle_length", "crop_cycle_length_days", "Cycle length", "days", "Water & calendar", True, ("irrigated",)),
)
DEFAULT_METRIC = "production_density"


def iter_metrics():
    """Yield normalized metric dicts from METRICS tuples."""
    for row in METRICS:
        metric_id, suffix, label, unit = row[0], row[1], row[2], row[3]
        group = row[4] if len(row) > 4 else "Potential"
        zero_is_missing = row[5] if len(row) > 5 else True
        water_modes = tuple(row[6]) if len(row) > 6 else WATER_MODES
        yield {
            "id": metric_id,
            "suffix": suffix,
            "label": label,
            "unit": unit,
            "group": group,
            "zero_is_missing": bool(zero_is_missing),
            "water_modes": list(water_modes),
        }


def metric_suffixes() -> tuple[str, ...]:
    return tuple(m["suffix"] for m in iter_metrics())



@dataclass(frozen=True)
class CropDefinition:
    crop: str
    label: str
    gaez_code: str
    gaez_variant_codes: tuple[str, ...] = ()

    @property
    def gaez_codes(self) -> tuple[str, ...]:
        return (self.gaez_code, *self.gaez_variant_codes)


CROPS = (
    CropDefinition("banana", "Banana", "BAN"),
    CropDefinition("barley", "Barley", "BRL"),
    CropDefinition("cassava", "Cassava", "CSV"),
    CropDefinition("chickpea", "Chickpea", "CHK"),
    CropDefinition("cowpea", "Cowpea", "COW"),
    CropDefinition("dry_pea", "Dry pea", "PEA"),
    CropDefinition("phaseolus_bean", "Phaseolus bean", "PHB"),
    CropDefinition("mung_bean", "Mung bean", "GRM"),
    CropDefinition("pigeonpea", "Pigeonpea", "PIG"),
    CropDefinition("soybean", "Soybean", "SOY"),
    CropDefinition("taro", "Taro", "TAROD", ("TAROW",)),
    CropDefinition("foxtail_millet", "Foxtail millet", "FML"),
    CropDefinition("maize", "Maize", "MZE"),
    CropDefinition("oats", "Oats", "OAT"),
    CropDefinition("pearl_millet", "Pearl millet", "PML"),
    CropDefinition("rice_dry", "Dry rice", "RCD"),
    CropDefinition("rice_wet", "Wet rice", "RCW"),
    CropDefinition("rye", "Rye", "RYE"),
    CropDefinition("sweet_potato", "Sweet potato", "SPO"),
    CropDefinition("sorghum", "Sorghum", "SRG"),
    CropDefinition("wheat", "Wheat", "WHE"),
    CropDefinition("white_potato", "White potato", "WPO"),
    CropDefinition("yam", "Yam", "YAM"),
)
CROP_BY_NAME = {crop.crop: crop for crop in CROPS}
# Locked FAO GAEZ v5 SHA256 digests (copied from ProsperOrPerishPopulationCapacityPipeline).
GAEZ_V5_YXX_SHA256 = {
    ("banana", "rainfed"): "7974e449008df30bc0a7392fdbe9247571490cdb9d71339a950f0b584a21b88d",
    ("banana", "irrigated"): "a4958d99ad7e78248926320ee04d34502b4e068cac10872a74c5951feadce20d",
    ("barley", "rainfed"): "7e0378c695260059655f339c286d84a51bf9ddacd2bd6e4a5073160f271f6e80",
    ("barley", "irrigated"): "3a878652a8c498a41ea07da016b3e344ec0ffa8f6ef44938a3141a5300698d56",
    ("cassava", "rainfed"): "926d201433ffaaf0aa4553a422b839105e68ad9ecef1562aec6d10b7687d74b6",
    ("cassava", "irrigated"): "35f6add900f3b86a6ca440da408cc5d4c68785654f76d48758aa14976be250d7",
    ("chickpea", "rainfed"): "e4e37b3a19c1ff4992ec6be494a1fecd2b033637530ea6cd990b11884d6a29b9",
    ("chickpea", "irrigated"): "fe3a9312f1346277d20d5354bb576dd37115cfff1ac9a3709fdd77e13e8f7ad0",
    ("cowpea", "rainfed"): "6026aae29c584d453715ad8832a39011c042f7207335de0aea93e3a72587cf6b",
    ("cowpea", "irrigated"): "6aa77e07c195cd4fda5dcc81738f997a83ff44d515eb01682bb25dda8499c6ff",
    ("dry_pea", "rainfed"): "66b337634ac2b13f3d696469e278b94bd692bca963b691a650c1953f6e5c0978",
    ("dry_pea", "irrigated"): "9bed2670c2b86145685ae77e15fef7070e3e06ca26ec5cec9b1a29428e3e30c3",
    ("phaseolus_bean", "rainfed"): "13d8d4ebb24538d74e2a45dd84ae73670857ed7347ffcaf9c360808ec26be4cd",
    ("phaseolus_bean", "irrigated"): "6333a8620bf0ffb92ad028422c0d148e2ccdbf88f85fe97170a463e087d59ff3",
    ("mung_bean", "rainfed"): "adb0878f8407ca4ee1fc9b5f441f36176903cc2b1e2dca8c9a88360a7c4a6e3b",
    ("mung_bean", "irrigated"): "30b9236abaccad0bca68cd8e57818074991c2738b2faa565eec554ed084c8958",
    ("pigeonpea", "rainfed"): "2e04c4431013d9d533dff4619050113775efb18afe3cf822dad97e2d9ec7e800",
    ("pigeonpea", "irrigated"): "1ce549b0a0ba9ddde60346fd59ca145e173222df9854d370eecb0085b4a58295",
    ("soybean", "rainfed"): "d8b414b987093aab024e516a3e9ba0e63e5be6e73908509a0c0d3dd4309a3d82",
    ("soybean", "irrigated"): "53eaf2c0e03f820742d85a2c8f8a0a0505d23667f9e711674ba5185fd20b4380",
    ("foxtail_millet", "rainfed"): "be0181b9ddc2e15d5622824d9f9f3ecdd3009966903067728a2c1f1857c7cf90",
    ("foxtail_millet", "irrigated"): "7d04e94e2bb73cb99129fa406677732d1f05d1bc4bad8a98d3465c2645854d57",
    ("maize", "rainfed"): "aa9a0d5e294ab5d3b602f8998087892f11887c32fb78c95b9f12d7b540cae2ab",
    ("maize", "irrigated"): "60a667e68d1ba64ccdecffc8add7df1a0379947ebc3701eea161a4e1d501bd58",
    ("oats", "rainfed"): "768bdc28482964743f2371ff8412db2aaf10993f8178e21f141a03f7fc586512",
    ("oats", "irrigated"): "d32ffa970ab5ddeff29b315a05c72678d2943f61fd7f4eaac6934da36806104a",
    ("pearl_millet", "rainfed"): "954caf1f9c3e33c47ba5d4aa34f915d7054539588887ae2c6fc01bd8815469d2",
    ("pearl_millet", "irrigated"): "82f1caca788d219e9249fc55087291d53d0faee93682def48793d324250a3a4f",
    ("rice_dry", "rainfed"): "7d47e4aa0ca1a637bc79d90f5bd3b8bfb7a1847f43b7845e6b9aef61454ef3fb",
    ("rice_dry", "irrigated"): "e5abc499d0a85f50d52facadbff076330f54c401e3f0cb7a905968788bd7647d",
    ("rice_wet", "rainfed"): "dcc26bca27055fcdb1951b292069df72ae1fd844dceaf700ac7917e7b3bd7d50",
    ("rice_wet", "irrigated"): "f3a00c9bf5c91969f1fda7de1ea78c39087cfda370a8c3c22d5ab12790292e1a",
    ("rye", "rainfed"): "f282930eac47eee0117e175abbd6b4b7167de5bcf431c14c7ed92795f4f5fac4",
    ("rye", "irrigated"): "6cc655b772bc5a800777580097ef3969a7e9cd739b25c78270f30783eb788912",
    ("sweet_potato", "rainfed"): "3cbf863e50173679af0bf9f67df875a48b137ed9c2107970ded25f071f76cde8",
    ("sweet_potato", "irrigated"): "eccff72a13a8f25559ff10379408f8395377ce2670f0233630814a672f9e3c0f",
    ("sorghum", "rainfed"): "e7ff14e71bd1512333be21a33892fb0882be860421ed8400cf0c3311b2d7a2f3",
    ("sorghum", "irrigated"): "25bb39a62959f226172920d45e8c5eacef4e2741b81cac440c41356331fd9eb9",
    ("wheat", "rainfed"): "1c21a10269833b97151576a7b67b14770461f2f4e2d2de0810a0fb418f3cfea6",
    ("wheat", "irrigated"): "3e8e94dbb0f87d8999aa59fc6b74ddfa1cc0b884e6773eb4eaf536ecd5cbc801",
    ("white_potato", "rainfed"): "10e25214abdfe93eedd4a930a9c63b29461229e0c0b2584c5dcc79f756f947b9",
    ("white_potato", "irrigated"): "a4688a90cdeb376295e2d7b2e1112703d1e43a5e078cfb5e14b405abe2d57513",
    ("yam", "rainfed"): "0fa67535efccd99c79719e25475bd9e9fe830d8c0abaa6f91dc8766b899acfdf",
    ("yam", "irrigated"): "d07c8144768deab52b15be1499b468c28bf0422ab6dc63c8f1aa7fd03f8c6c6c",
}

GAEZ_V5_YLX_SHA256 = {
    ("banana", "irrigated"): "d4f794fa7ac9197775167e6a78a3c3191927a0e5d49fcb3b372f77c410f08bea",
    ("banana", "rainfed"): "6dcb71b4a9e4f20080043f7582aabd6df26af379e1b2ca739c069c315658ea44",
    ("barley", "irrigated"): "903221315af2b1d104ad5ecf46c2dce4d9df82e682a4d8f69012c517cce3d985",
    ("barley", "rainfed"): "3ef72884feb7e17554ccde5b3f6843b3ec8f503a1fabd7de4388dd2407b1e675",
    ("cassava", "irrigated"): "bd8adf48c58de082a86f08f2e41cdc075663c796cc4b796e52333563bb1376b5",
    ("cassava", "rainfed"): "586f88d972d7edc76cc67ebd1f2188e3a69d24050075c71a1073be5abc7a4e0b",
    ("chickpea", "irrigated"): "07bddd38fed9a37ea78f387b06ad0a472d090d172cd8830bca4820155bc483a4",
    ("chickpea", "rainfed"): "141ab2207f857438a2db2600a4a64df2286e3b1a2ecc962c645cb2d0198453b8",
    ("cowpea", "irrigated"): "87a90760894d48be1b53bbbe0cdd290693b4195fb73512aa54705a507636be73",
    ("cowpea", "rainfed"): "19714cc7b5a1661a9aeba61554db15c61f3fef7db944fc1bb85035fbab225921",
    ("dry_pea", "irrigated"): "ddfe4340197109f37541b97c0098d61a18d1c0d34693f8077073253c803e209e",
    ("dry_pea", "rainfed"): "9b136f6d4f6f417f06834902c67d18a715f513e7a8e5ee5875dd269a3ba5f6ca",
    ("phaseolus_bean", "irrigated"): "45c27c0aad667f38c51b5c52c761aff6c236a963edd74252bfa6d14f25c7c295",
    ("phaseolus_bean", "rainfed"): "3e433b59c41cfa9fa443e830ff4458efcb3fc53a91ebf38c075062fc2c74f949",
    ("mung_bean", "irrigated"): "073bdc04f688fad3ffcd679d54eb4452eb806c0b7624cae4d144649dae95dd36",
    ("mung_bean", "rainfed"): "8f5c2310af458bd329fd7dd9d15705ee03ae009bdd21ce0729ff28581a59db28",
    ("pigeonpea", "irrigated"): "bf26d233e0bf9c6dd4a9c02117be845339ecb66840e8ac0c2695e8df15f8b4de",
    ("pigeonpea", "rainfed"): "410fc4f5e68ddc2b09083d4cdd2a9b9730e03e88b4b412077822922efd3024c9",
    ("soybean", "irrigated"): "092198784f302d48fc8b1e26ef24e66f013f83acc96bec105ae94d695344eb68",
    ("soybean", "rainfed"): "55bd711360ca2047494e1f68f9a0b1d945b1ac8f7907fe1417016e5b1c06eca9",
    ("foxtail_millet", "irrigated"): "10280a58ee82051b219b9db2ce65b725790d9f1a09f76804c9a920d2747e6e92",
    ("foxtail_millet", "rainfed"): "3dc82734e856e4a7ecd2a8461b2e40e9e43eb4e8e7b0309172f03dafe67d84d0",
    ("maize", "irrigated"): "c41dad13af515ccdc2c4fe5cd2824590e7d8626365c8fdb7d8a21508f87c647b",
    ("maize", "rainfed"): "2880d15aff4beba157861013843d0f1dec7efe88e97d2dba7d0576b44901bb89",
    ("oats", "irrigated"): "1a39fd8476912bd493c58da30c7f3270cd43596de956125835bd6eca2f77f5dc",
    ("oats", "rainfed"): "0d785776c6748825e96dc4be208b0a06a8fc2d8cdd03fd18dbe50782f4da0648",
    ("pearl_millet", "irrigated"): "cc3272c8ea4d386927d882f82a809a1836e187a114cd22102312a386f726edf7",
    ("pearl_millet", "rainfed"): "b1078b5f029f4d057d3279f78014abc751926b2372d8077ce947f20caa8bd6f6",
    ("rice_dry", "irrigated"): "67576ea52b9dd43856bd7162bbe4914b7e9c2841e93e8643c60e620b3678feec",
    ("rice_dry", "rainfed"): "a144e61fdafebf06ef05f674e192262169de900f213e5b576c1e6d997df3139b",
    ("rice_wet", "irrigated"): "a1b6cfafe81d702d9e8b4a9a01ea8866b40d2750e8bf6a1deae428b422b13044",
    ("rice_wet", "rainfed"): "dcadae554e97976be04af16874604970352686d14241184e4af4c2b63ee2b6b3",
    ("rye", "irrigated"): "34ca5554da2c0ae255cd163df77a7e0113494107a567ef8356cbe46717537c89",
    ("rye", "rainfed"): "a9d28a6b47ed9bb367dca9afe3a71fe4e0d4baa1170cebcee6c74b308ddc627c",
    ("sorghum", "irrigated"): "ffb981093aec0795089e228e188cdf716514d17e1b3e6688cb53b4a58fd56937",
    ("sorghum", "rainfed"): "5977b3e6adede651ff78d81912f6d9900ccbe3598ddc239ee2a92f18324616f6",
    ("sweet_potato", "irrigated"): "9f10fda845c149f181b1f0c3bd415f7a30c8f6f77c743c049aecffe7f140f156",
    ("sweet_potato", "rainfed"): "9899c93f2f79bb49c1464d4ff2f2529bf9f99c1ce4d7f13a288187c2827ce5da",
    ("wheat", "irrigated"): "c399920e43379893368c154c58ced7dad53feea8531eef5cf772b5f63eaf7473",
    ("wheat", "rainfed"): "b5d7b469dabcbaefe8fcecd2fbce11d344e545c3679eb3ab416b13e1524a8dc8",
    ("white_potato", "irrigated"): "56845087fe7cd4a02d0affff7aa7e28bd2d6186513de77c05cdbbfb8ce3d1d28",
    ("white_potato", "rainfed"): "1ace43adeee6bd79ec001717cf4a581c22180c15e37e77b2b0a9dd52a61e953e",
    ("yam", "irrigated"): "ad1430bf9cb250f85180274a3b282821977fd5a0b96d3c4b931145e8a89516d5",
    ("yam", "rainfed"): "27da0381da744a04fdc0a298971ef15cfe53ebfca1b11435aafe2c12b17c36de",
}

GAEZ_V5_SX3_SHA256 = {
    ("banana", "irrigated"): "bf26236fecf3e7e2720e2a1d9dc71c1854f243f49b527475710a34955078088e",
    ("banana", "rainfed"): "2d3810e5012e4150f36a4dd88b098e301ec75f7c8082031f9378ae805a3c402b",
    ("barley", "irrigated"): "84f6a42e3fb415830933683220a0f16230e341f99f002ebeca8b623e354dc3c1",
    ("barley", "rainfed"): "8dbe8d8eae47c5a7890965b1a7984ebcc6ac2dff4b829fcad0e1c69fae994944",
    ("cassava", "irrigated"): "9c835e2c870e0dec209d7074af9c92031a60f390780e720fb7e90d992beb11e7",
    ("cassava", "rainfed"): "e8e362be53a57a3062280684694b7b82fa81030ab0bc520179cb402d67077638",
    ("chickpea", "irrigated"): "29cefc45cda85b8cccbddb5d86ad69a61a43ef8afa62d1e1348a53bd9922a5d9",
    ("chickpea", "rainfed"): "f28e05004425996938114a9403a77cb981e4a7b3467ce38c8c93f6eb3db8acb2",
    ("cowpea", "irrigated"): "bca6c3b6584a9ffbfa1d4d0a532aec051470ca80e83783cf00f8b8b4aa431e51",
    ("cowpea", "rainfed"): "fad116a76b4a75916b63fb38090567082528288e5b86083ed4fdf47b5d9fd35c",
    ("dry_pea", "irrigated"): "e07d704b33165dbd6f120ec8bd9874169b88878c46d04a00858a5fd8461ea203",
    ("dry_pea", "rainfed"): "8e146439b115d40e8a099ab45ec3b35c9e733ddd073ca0e25403d4123c6f10c3",
    ("phaseolus_bean", "irrigated"): "756c4f146e9f6f40523ec4da641ac4346d64ed8101662e6848996dd869962990",
    ("phaseolus_bean", "rainfed"): "ebc8fc96f643b2951eaa50cb29a7b0014aea02bbf425f3faf8c12abbfab66ac8",
    ("mung_bean", "irrigated"): "ac1d00e74c45038b7a45565a611dde5c78475b53206fa601d04d7477f7b5308c",
    ("mung_bean", "rainfed"): "30d1160b353784a3ddbb8f03d2dd468f88ac84b364459e61035face1885081ba",
    ("pigeonpea", "irrigated"): "a6554c877dcb8c53b65187e7f0bf25028be74b385c9297229805bcb2f73f86ca",
    ("pigeonpea", "rainfed"): "efa9ce4d67a396399301660cd8279b022f5c5eb60643c8b130577650ec24b099",
    ("soybean", "irrigated"): "5419ca2cadffa558ad59f0520ee41512e973183465a0bb7e6f96f4f7ea88f975",
    ("soybean", "rainfed"): "54556d5beaf5fbd524625217c0841361ec33b4ba339b7be0c352dcfe608dcfae",
    ("foxtail_millet", "irrigated"): "ac3ecd5f5079820d30f52b12224fd35612966a7322a4350daacdedf93ff983da",
    ("foxtail_millet", "rainfed"): "1d6018fe941735a006f01dd57f35faa91743b8c4a14c0b7e7142144a5a98d77d",
    ("maize", "irrigated"): "4e8ab40d1eac09ebf9d11f87d59cb60b5bc7308322b2a3b857eb2a29d8ba944e",
    ("maize", "rainfed"): "5ceba693ceefb1bc91af354d1799d51c946c84e7b5d2ecfeb310be1d9b12637c",
    ("oats", "irrigated"): "4d04607291923adc29d0eaa406f9b069b7922750863642e45d9b7c18d2b0d3c2",
    ("oats", "rainfed"): "ab7f7483ab492c3a242e1b5c1566a8ca539e6cf4dfdb91c60bee17e1794b6acd",
    ("pearl_millet", "irrigated"): "03352a40cc9ed44e98d316f9dcf1531ce73bc913241dab4b1361f65da5de4b5f",
    ("pearl_millet", "rainfed"): "97ccc8924b1eef12e3c56d7b2a3a12c9c7daf360dea5fe605067ba7688133cb5",
    ("rice_dry", "irrigated"): "2d22b1671e66bf176f92b1b3d1abe20350b8fb5b230d19116c5d32fdd5fe2264",
    ("rice_dry", "rainfed"): "845e8dfb4cd841ae8b9b9a9a78bbdeefce54dff6ad01f9ed0e24540bc13c9bee",
    ("rice_wet", "irrigated"): "9eb8c66102ebbd89b5a776a12645bb7b3a9f22547ace8c0c895826eb92962af6",
    ("rice_wet", "rainfed"): "43005117605f481116b110ea9c1ea9aed77347a2d74cc08ce2476822a3cb877c",
    ("rye", "irrigated"): "7f63047d75517afd8efe2a24a708f52e962a29b92d05d4414d49cff75bcc4864",
    ("rye", "rainfed"): "1b86c3388accca1f618cc892e036c18c9f0d776068f41d64687bba9e9c6fce2d",
    ("sorghum", "irrigated"): "ea72c3e92ff9e19038e635bd6149aa8ee60bf7cef1a696198fa1bbe9451ae0cc",
    ("sorghum", "rainfed"): "2726b3605eb6871a2c2df0c3ef9f3517783d85bc5133ff746efa9f7c9a769193",
    ("sweet_potato", "irrigated"): "51b73e0f25db3c56234019996f3ceeacff74bfcf4e6474cc9a82391ecde899ab",
    ("sweet_potato", "rainfed"): "ae2d125241fbd45781d1d8e8e89854641923da7224c2186f63de6a11dbdd644e",
    ("wheat", "irrigated"): "e025a58c7bbe3656783c6b6ddf25ccc6efae9065f7df531aa4c58ab4b86b1fb3",
    ("wheat", "rainfed"): "c5c1bce2e3fc0451d403ca20b93576b88a00a8c42d9d5d60271a43012a5625d5",
    ("white_potato", "irrigated"): "74ce65c51031787a72e05edc594fe7f401b196b7fcb59793f76acacf54b1d961",
    ("white_potato", "rainfed"): "80c3b2fef6b3d8262e4ebdee9a1182fe3cbff80f37619f4fe34008899550efa4",
    ("yam", "irrigated"): "d22a5f6b4321a623aa78a8903b245008cfaac44fb0308e2f45546f11f994474d",
    ("yam", "rainfed"): "459b8d0a96854e890962465e2e2a0bffe1cfea8cf15cd839db786310104f1dc8",
}

GAEZ_V5_VARIANT_SHA256 = {
    ("RES05-YXX", "taro", "rainfed", "TAROD"): "661b498a4c04fb43e25bb421e0fae2e5870ff496e890cb65e1045a05fdc075df",
    ("RES05-YXX", "taro", "irrigated", "TAROD"): "7a9b0784dd65ea5bb61a96924b018fb1dd5aa5adddf27a18b2f1b4b61a0a9d7c",
    ("RES05-YXX", "taro", "rainfed", "TAROW"): "1b0489760d3a76449f958a58c9cfa56636caddf289d9ef8503cc40b178db4421",
    ("RES05-YXX", "taro", "irrigated", "TAROW"): "0ff065eb948f8378f2dd321dee5fdc8d30efb3b326faa705823a7667e6a6614a",
    ("RES05-YLX", "taro", "rainfed", "TAROD"): "fe9967f9d83a5059b8c7d96959b94140b7dadfa933b8a51b99fb994196465b11",
    ("RES05-YLX", "taro", "irrigated", "TAROD"): "f23fef0d8f404231431470990799e27c6875e250654f10635fb656968b4d71fd",
    ("RES05-YLX", "taro", "rainfed", "TAROW"): "fa8b5db43955721ef215ce4c79d9e85aa61f9188e76c2c123bd40d60e01a8999",
    ("RES05-YLX", "taro", "irrigated", "TAROW"): "9bdfb0b97fd824dd2a1224c6390e8c9f357edc1ca2744a177454bb7ca6437869",
    ("RES05-SX3", "taro", "rainfed", "TAROD"): "58d2b2a662e813562d87d3d8137d76749632a9792cfc94e8a87c64e51454eb81",
    ("RES05-SX3", "taro", "irrigated", "TAROD"): "be456fb149c852bc7e188979e7e1964f2c504d77e1026e9194966c523f91630c",
    ("RES05-SX3", "taro", "rainfed", "TAROW"): "a18e0edaad9ef2a7caea9e8c06f47bb419b4ac30ef314ff38a069082b966a75d",
    ("RES05-SX3", "taro", "irrigated", "TAROW"): "cce66b5213f73fd088b4499b895e2ce039ebd9d121c8df7baeb541aa3b199496",
}


@dataclass(frozen=True)
class RasterSpec:
    crop: str
    crop_code: str
    crop_variant: str
    variable: str
    water_mode: str
    management_code: str
    filename: str
    url: str
    expected_sha256: str
    cache_relpath: str


def management_code(water_mode: str) -> str:
    if water_mode == "rainfed":
        return "LRLM"
    if water_mode == "irrigated":
        return "LILM"
    raise ValueError(f"unsupported water mode: {water_mode}")


def crop_variant(crop: str, crop_code: str) -> str:
    if crop != "taro":
        return "standard"
    return {"TAROD": "dryland", "TAROW": "wetland"}[crop_code]


def expected_sha256(variable: str, crop: str, water_mode: str, crop_code: str) -> str:
    variant = GAEZ_V5_VARIANT_SHA256.get((variable, crop, water_mode, crop_code))
    if variant:
        return variant
    if len(CROP_BY_NAME[crop].gaez_codes) > 1:
        return ""
    if variable == "RES05-YXX":
        return GAEZ_V5_YXX_SHA256.get((crop, water_mode), "")
    if variable == "RES05-YLX":
        return GAEZ_V5_YLX_SHA256.get((crop, water_mode), "")
    if variable == "RES05-SX3":
        return GAEZ_V5_SX3_SHA256.get((crop, water_mode), "")
    if variable == "RES05-SIX":
        # Suitability-class rasters are unlocked; verify by successful download/read.
        return ""
    raise ValueError(f"unsupported GAEZ variable: {variable}")


def metric_column(crop: str, water_mode: str, metric_suffix: str) -> str:
    return f"{crop}_{water_mode}_{metric_suffix}"


def selected_crops(names: list[str] | None) -> tuple[CropDefinition, ...]:
    if not names:
        return CROPS
    missing = [name for name in names if name not in CROP_BY_NAME]
    if missing:
        raise ValueError(f"unknown crops: {', '.join(missing)}")
    return tuple(CROP_BY_NAME[name] for name in names)


def raster_specs(
    *,
    crops: tuple[CropDefinition, ...] | None = None,
    water_modes: tuple[str, ...] = WATER_MODES,
    base_url: str = GAEZ_V5_BASE,
    variables: tuple[str, ...] = ("RES05-YXX", "RES05-YLX", "RES05-SX3"),
) -> list[RasterSpec]:
    crops = crops or CROPS
    mapset_root = base_url.rsplit("/", 1)[0]
    specs: list[RasterSpec] = []
    for crop in crops:
        for water_mode in water_modes:
            mgmt = management_code(water_mode)
            for crop_code in crop.gaez_codes:
                variant = crop_variant(crop.crop, crop_code)
                for variable in variables:
                    filename = (
                        f"GAEZ-V5.{variable}.HP8100.AGERA5.HIST."
                        f"{crop_code}.{mgmt}.tif"
                    )
                    short = variable.removeprefix("RES05-").lower()
                    folder = f"res05_{short}"
                    url = (
                        f"{base_url}/{filename}"
                        if variable == "RES05-YXX"
                        else f"{mapset_root}/{variable}/{filename}"
                    )
                    specs.append(
                        RasterSpec(
                            crop=crop.crop,
                            crop_code=crop_code,
                            crop_variant=variant,
                            variable=variable,
                            water_mode=water_mode,
                            management_code=mgmt,
                            filename=filename,
                            url=url,
                            expected_sha256=expected_sha256(
                                variable, crop.crop, water_mode, crop_code
                            ),
                            cache_relpath=str(
                                Path("gaez_v5")
                                / folder
                                / "HP8100"
                                / "AGERA5"
                                / "HIST"
                                / mgmt
                                / filename
                            ),
                        )
                    )
    return specs


def sha256_lock_count() -> dict[str, int]:
    return {
        "yxx": len(GAEZ_V5_YXX_SHA256),
        "ylx": len(GAEZ_V5_YLX_SHA256),
        "sx3": len(GAEZ_V5_SX3_SHA256),
        "variant": len(GAEZ_V5_VARIANT_SHA256),
        "crops": len(CROPS),
    }
