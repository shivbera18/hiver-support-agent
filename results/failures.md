# Failure mining (author: pick top-5 with real examples)

## Top intent confusions
intent_gold             intent                
other_offtopic          billing_refund            7
                        setup_howto               5
                        hardware_damage_repair    5
software_update         hardware_damage_repair    4
hardware_damage_repair  other_offtopic            3

## Low judge scores (overall<=2)
- id=146585 gold=hardware_damage_repair pred=other_offtopic cust=I have a problem with my earpods...this is my second pair that faces the same problem for the record. The right earbud stopped working all o reply=@236219 You’ve come to the right place, and we’ll be happy to look into this with you. To begin, which iOS version is installed on your iPho
- id=2919996 gold=software_update pred=setup_howto cust=@115858 @115890 Apple, why is this happening on my MacBook Pro after installing an update and how do I fix it? <URL> reply=@314749 Is this happening on the same network?
- id=2505744 gold=other_offtopic pred=setup_howto cust=how do I find my booking? reply=@294225 We'd like to help! Check out this article: https://t.co/G7BHuQnDS6
- id=2468992 gold=other_offtopic pred=setup_howto cust=So the answer is "No, GWR are not investing in maintenance and repair". Seems short sighted, even if new rolling stock is available. If the  reply=@767530 We are here to help. While they are a different subscription, songs from your library will still be uploaded and matched.
- id=1868546 gold=other_offtopic pred=setup_howto cust=I was supposed to be on hold for 2 mins its been 20 and I need to leave. How do i reschedule call reply=@476539 That's not the experience we'd like you to have. We'd be happy to help, send us a DM with more info. https://t.co/GDrqU22YpT

## Hallucinated=yes
