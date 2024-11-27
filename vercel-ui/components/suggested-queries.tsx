import { motion } from "framer-motion";
import { Button } from "./ui/button";

export const SuggestedQueries = ({
  handleSuggestionClick,
}: {
  handleSuggestionClick: (suggestion: string) => void;
}) => {
  const suggestionQueries = [
    {
      desktop: "What is the distribution of our customers who have paid, paused and do not have subscription?",
      mobile: "Subscription Distribution",
    },
    {
      desktop: "What is the total billing amount per month, adjusted for currency exchange rates, from September 2022 to October 2023?",
      mobile: "Total Billing Amount",
    },
    {
      desktop: "What are the customer creation times and dates in the 'America/Los_Angeles' timezone for March 2023?",
      mobile: "Customer Creation Time",
    },
    {
      desktop:
        "What is the total billing amount and number of outstanding invoices that are not paid or payment is due?",
      mobile: "Total outstanding amount",
    },
    {
      desktop: "Which subscription plans have the highest number of active customers?",
      mobile: "Subscription with active status",
    },
    {
      desktop:
        "What are the quarterly trends in Paid LTV based on subscription status changes?",
      mobile: "Quaterly Trends",
    },
    {
      desktop: "Identify the subscription status changes each quarter, such as cancellations, upgrades, and downgrades.",
      mobile: "Subscription status",
    },
  ];

  return (
    <motion.div
      key="suggestions"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      layout
      exit={{ opacity: 0 }}
      className="h-full overflow-y-auto"
    >
      <h2 className="text-lg sm:text-xl font-semibold text-foreground mb-4">
        Try these queries:
      </h2>
      <div className="flex flex-wrap gap-2">
        {suggestionQueries.map((suggestion, index) => (
          <Button
            key={index}
            className={index > 5 ? "hidden sm:inline-block" : ""}
            type="button"
            variant="outline"
            onClick={() => handleSuggestionClick(suggestion.desktop)}
          >
            <span className="sm:hidden">{suggestion.mobile}</span>
            <span className="hidden sm:inline">{suggestion.desktop}</span>
          </Button>
        ))}
      </div>
    </motion.div>
  );
};
