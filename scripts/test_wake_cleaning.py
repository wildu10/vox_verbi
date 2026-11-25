import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from command_router import match_and_strip_wake_word, CommandRouter, normalize_text
from pre_filter import PreFilter

def test():
    router = CommandRouter(canonical_json_path="data/canonical_test.json")
    prefilter = PreFilter()
    
    test_cases = [
        "Busverby, quien ganó el partido de fútbol ayer.",
        "Vos ver mi, la cruz que nadie quería.",
        "Vos ver mi, ¿Quién es Jesús?",
        "Busverby, al paraíso",
        "BusBerby. Llevamos...",
        "Vox Verbi, quiero confesar mis pecados contigo.",
        "Vox Verbi, ¿quién ganó el partido de fútbol ayer?",
        "Bosberg, que enseña de i verbum sobre la revelación.",
        '«"Bos verbi" que enseña "dei" verbun sobre la revelación»',
        "Vox Verbi, ¿qué me quiere decir Dios con el sueño que tuve anoche?",
        "Vox Verbi, ¿qué quiere Dios para mi futuro?",
        "Vox Verbi, decisión quiere Dios que tome mañana sobre mi trabajo?",
        "que enseña de iverbum sobre la revelación",
        "Vox Verbi, al paraíso",
        "Vox Verbi, finalizamos la adoración"
    ]
    
    for tc in test_cases:
        matched, rest = match_and_strip_wake_word(tc)
        real_words = [w for w in re.sub(r'[^\w\s]', '', rest).split() if len(w) > 1]
        direct_route = router.route(rest) or router.route(f"vox verbi {rest}")
        act = direct_route.get("action") if direct_route else None
        
        # Test prefilter
        allowed, literal = prefilter.evaluate(rest or tc)
        
        print(f"INPUT: '{tc}'")
        print(f"  Matched: {matched}")
        print(f"  Rest: '{rest}'")
        print(f"  Real words count: {len(real_words)}: {real_words}")
        print(f"  Direct route action: {act}")
        print(f"  Prefilter: allowed={allowed}, literal='{literal}'")
        print("-" * 50)

if __name__ == "__main__":
    test()
