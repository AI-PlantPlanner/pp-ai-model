"""한국어 조사(은/는, 이/가) 자동 선택.

식물 이름을 문장 중간에 끼워 넣을 때, 이름 마지막 글자에 받침이 있는지에 따라
조사가 달라집니다(예: "송악은" vs "테이블야자는"). 이걸 문장 템플릿마다 손으로
다 맞춰 쓰면 실수하기 쉬워서, 이름을 보고 자동으로 올바른 조사를 골라주는
헬퍼입니다.
"""


def has_final_consonant(word: str) -> bool:
    """단어의 마지막 글자에 받침이 있으면 True.

    한글 완성형 글자는 유니코드에서 0xAC00~0xD7A3 범위에 순서대로 배열되어 있고,
    (코드 - 0xAC00) % 28 이 0이면 받침이 없는 글자입니다(0이 "받침 없음"에 해당).
    한글이 아닌 문자로 끝나면(영문/숫자/특수문자 등) 안전하게 받침이 있는 것으로
    취급합니다 — "은"/"이" 계열이 어색하게 들리는 경우가 "는"/"가" 계열보다 적어서요.
    """
    if not word:
        return False
    code = ord(word[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return True


def eun_neun(word: str) -> str:
    """'은' 또는 '는' 중 word에 맞는 걸 반환. 예: 송악 -> 은, 테이블야자 -> 는."""
    return "은" if has_final_consonant(word) else "는"


def i_ga(word: str) -> str:
    """'이' 또는 '가' 중 word에 맞는 걸 반환."""
    return "이" if has_final_consonant(word) else "가"
