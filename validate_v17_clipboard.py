import argparse,json,pyperclip
from profit_taker.axiom_clipboard import clipboard_looks_like_axiom,parse_clipboard_cards
ap=argparse.ArgumentParser(); ap.add_argument("--json",action="store_true"); a=ap.parse_args(); text=pyperclip.paste() or ""; cards=parse_clipboard_cards(text)
out={"valid_axiom_selection":clipboard_looks_like_axiom(text),"cards": [c.to_dict() for c in cards]}
print(json.dumps(out,indent=2,default=str) if a.json else out)
