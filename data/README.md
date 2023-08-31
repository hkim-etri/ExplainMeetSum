This is the example( `covid10.json` ) of dataset format

```
{
    "domain": "Committee",
    "topic_list": [
        {
            "topic": "Introduction of petitions and prioritization of governmental matters",
            "relevant_text_span": [["0","19"]]
        },
        {
            "topic": "Financial assistance for vulnerable Canadians during the pandemic and beyond",
            "relevant_text_span": [["21","57"],["113","119"],["191","217"]]
        },
        ...
    ],
    "explainable_ami": null,
    "explainable_icsi": null,
    "explainable_qmsum": {
        "general_query_list": [
            {
                "query": "Summarize the whole meeting.",
                "answer": "The meeting of the standing committee took place to discuss matters pertinent to the Coronavirus pandemic. The main issue at stake was to ...",
                "explainable_answer": [
                    {
                        "answer_sentence": "The meeting of the standing committee took place to discuss matters pertinent to the Coronavirus pandemic.",
                        "evidence": [
                            {
                                "type": "CES",
                                "turn_index": 0,
                                "sent_index": 1,
                                "speaker": "The Chair (Hon. Anthony Rota (NipissingTimiskaming, Lib.))",
                                "content": "Welcome to the third meeting of the House of Commons Special Committee on the COVID-19 Pandemic."
                            },
                            {
                                "type": "CES",
                                "turn_index": 0,
                                "sent_index": 5,
                                "speaker": "The Chair (Hon. Anthony Rota (NipissingTimiskaming, Lib.))",
                                "content": "Colleagues, we meet today to continue our discussion about how our country is dealing with the COVID-19 pandemic."
                            }
                        ],
                        "answer_sentence_index": [
                            0
                        ]
                    },
                    ...
                ]
            },
            ...
        ],
        "specific_query_list": [
            {
                "query": "Summarize the discussion about introduction of petitions and prioritization of government matters.",
                "answer": "The Chair brought the meeting to order, announcing that the purpose of the meeting was to discuss COVID-19 's impact on Canada. Five petitions were presented ...",
                "relevant_text_span": [["0","19"]],
                "explainable_answer": [
                    {
                        "answer_sentence": "The Chair brought the meeting to order, announcing that the purpose of the meeting was to discuss COVID-19 's impact on Canada.",
                        "evidence": [
                            {
                                "type": "CES",
                                "turn_index": 0,
                                "sent_index": 0,
                                "speaker": "The Chair (Hon. Anthony Rota (NipissingTimiskaming, Lib.))",
                                "content": "I call the meeting to order."
                            },
                            {
                                "type": "CES",
                                "turn_index": 0,
                                "sent_index": 1,
                                "speaker": "The Chair (Hon. Anthony Rota (NipissingTimiskaming, Lib.))",
                                "content": "Welcome to the third meeting of the House of Commons Special Committee on the COVID-19 Pandemic."
                            }
                        ],
                        "answer_sentence_index": [
                            0
                        ]
                    },
                    ...
                ]
            },
            ...
        ]
    },
    "meeting_transcripts": [
        {
            "speaker": "The Chair (Hon. Anthony Rota (NipissingTimiskaming, Lib.))",
            "content": "I call the meeting to order.  Welcome to the third meeting of the House of Commons Special Committee on the COVID-19 Pandemic ...",
            "sentence_level_content": [
                {
                    "turn_index": 0,
                    "sent_index": 0,
                    "dialogue_sentence": "I call the meeting to order.",
                    "dialogue-act_id": null
                },
                {
                    "turn_index": 0,
                    "sent_index": 1,
                    "dialogue_sentence": "Welcome to the third meeting of the House of Commons Special Committee on the COVID-19 Pandemic.",
                    "dialogue-act_id": null
                },
                {
                    "turn_index": 0,
                    "sent_index": 2,
                    "dialogue_sentence": "Pursuant to the order of reference of Monday, April20, the committee is meeting for the purposes of considering ministerial announcements, allowing members to present petitions, and questioning ministers of the crown, including the Prime Minister, in respect of the COVID-19 pandemic.",
                    "dialogue-act_id": null
                },
                {
                    "turn_index": 0,
                    "sent_index": 3,
                    "dialogue_sentence": "I understand there's an agreement to observe a moment of silence in memory of the six members of the Canadian Armed Forces who lost their lives last Wednesday in a helicopter crash off the coast of Greece.",
                    "dialogue-act_id": null
                },
                ...
            ]
        },
        ...
    ]
}
```